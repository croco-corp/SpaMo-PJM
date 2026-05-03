# Deadlock w `derangement()` — diagnoza i fix (2026-04-30)

## TL;DR

Trening `pretrain_v22c_phoenix_paper_dual` zawiesił się **przed pierwszą iteracją** (Lightning fit_loop wszedł, ale `training_step` nigdy się nie zakończył). Główny wątek 6h 44min kręcił się w nieskończonej pętli rejection sampling w `utils/helpers.py:derangement()`. GPU 0% mimo 14.8GB alokacji, wszyscy DataLoader workers spali bo nikt nie poprosił o następny batch.

**Root cause:** `derangement(lst)` używa rejection sampling po **wartościach** elementów. Jeśli `lst` zawiera duplikaty na tyle, że żadna permutacja nie spełnia warunku `lst[i] != shuffled[i] ∀i` — pętla `while True` nigdy nie wychodzi.

## Dowód (py-spy dump)

```
Process 1948888: python main.py -c configs/pretrain_v22c_phoenix_paper_dual.yaml ...

Thread 0x77B8E6F9E740 (active+gil): "MainThread"
    shuffle (random.py:389)
    derangement (helpers.py:18)
    get_inputs (t5_slt.py:453)
    training_step (t5_slt.py:692)
    training_step (pytorch_lightning/strategies/strategy.py:391)
    ...
    fit (pytorch_lightning/trainer/trainer.py:584)
    main (main.py:374)
```

Pozostałe 30 wątków: wszystkie na `futex_wait_queue` (śpią). DataLoader workers (8 osobnych procesów): wszystkie `Sl`, 0% CPU. GPU: utilization 0%, power 22.8W (idle), pamięć 14.8GB alokowana ale martwa.

`/proc/1948888/status`:
- `State: R (running)`, `wchan: 0` → main thread w userspace, busy-loop pythonowy.
- `Threads: 31`, `VmRSS: 2.76 GB`.

## Kod problemu

### `utils/helpers.py:12-20`

```python
def derangement(lst):
    if len(lst) <= 1:
        return lst

    while True:
        shuffled = lst[:]
        random.shuffle(shuffled)
        if all(original != shuffled[i] for i, original in enumerate(lst)):
            return shuffled
```

### Wywołanie w `spamo/t5_slt.py:453`

```python
ex_lang_translations = derangement(ex_lang_translations)
```

`ex_lang_translations` to lista tłumaczeń (jedno na próbkę w batchu) używanych jako negative samples w cross-lingual contrastive loss. Cel `derangement`: pomieszać listę tak, by każda próbka dostała tłumaczenie nie-swoje.

## Dlaczego pętla może się nie kończyć

Algorytm sprawdza warunek na **wartościach**, nie na **indeksach**. Klasyczny derangement matematyczny operuje na pozycjach (`σ(i) ≠ i`) i istnieje dla każdego `n ≥ 2`. Tutaj sprawdzane jest `lst[i] ≠ shuffled[i]` — czyli wartość w pozycji `i` po przemieszaniu nie może być równa wartości oryginalnej w pozycji `i`. Przy duplikatach to staje się problemem:

| `lst`             | Możliwy derangement po wartościach? |
|-------------------|--------------------------------------|
| `["A","B","C"]`   | tak (`["B","C","A"]`)                |
| `["A","A","B"]`   | **nie** (pozycje 0 i 1 wymagają nie-A; jest tylko 1 nie-A) |
| `["A","A"]`       | **nie**                              |
| `["", "", "X"]`   | **nie** (puste stringi się duplikują) |

Reguła ogólna (Hall): jeśli jakaś wartość pojawia się w więcej niż `n/2` pozycjach, derangement po wartościach jest niemożliwy.

## Najprawdopodobniejsza przyczyna w tym konkretnym batchu

Konfig: `pretrain_v22c_phoenix_paper_dual.yaml` z tagami `de-de paper-faithful`. Trzy hipotezy:

1. **Puste `ex_lang_translation`** — niektóre próbki nie mają cudzojęzycznego tłumaczenia, więc trafia tam `""`. Jeśli ≥2 próbki mają `""` i nie ma jednej z różną wartością — derangement po wartościach niemożliwy.
2. **Duplikaty po dataset deduplication** — phoenix-2014t ma powtarzające się zdania (krótkie prognozy pogody — wiele wariantów `"morgen kalt"` itp.). W batchu losowo trafiło ≥2 takich.
3. **Tag `de-de`** sugeruje source DE → target DE. Jeśli `ex_lang` w tym trybie zwraca tę samą wartość co `text`, lub same DE-DE pary się duplikują w batchu.

## Fix

Plan: zmienić `derangement` tak, żeby (a) miał limit prób, (b) miał deterministyczny fallback, który zawsze daje permutację bez fixed-pointów na **indeksach**. To zachowa intencję (każdy sample dostaje cudze tłumaczenie) i nigdy się nie zawiesi.

```python
def derangement(lst):
    if len(lst) <= 1:
        return lst
    n = len(lst)
    indices = list(range(n))
    for _ in range(1000):
        random.shuffle(indices)
        if all(indices[i] != i for i in range(n)):
            return [lst[i] for i in indices]
    # fallback: cyclic shift — gwarantuje brak fixed-points na indeksach
    return lst[1:] + lst[:1]
```

Zmiana:
- Warunek `indices[i] != i` zamiast `lst[i] != shuffled[i]` — operuje na pozycjach. Zawsze rozwiązywalne dla `n ≥ 2` (probability of success on random permutation ≈ 1/e).
- Cap 1000 prób + fallback na cyclic shift — nawet w teoretycznie złym losowaniu się skończy.
- Konsekwencja semantyczna: jak w batchu były duplikaty wartości, sample może dostać tłumaczenie identyczne ze swoim **przypadkiem** (gdy duplikat siedzi w innej pozycji). Dla negative samplingu to niegroźne i statystycznie rzadkie.

## Lessons learned

1. **`while True:` z wewnętrznym warunkiem rejection sampling** to zawsze podejrzany pattern — wymaga dowodu że warunek MUSI się spełnić w skończonym czasie. Tu go brakowało.
2. **Faulthandler** powinien być zarejestrowany w każdym długim treningu — `kill -USR1` zrzuciłby Python stack do loga bez ptrace, więc wykrylibyśmy to za 10 minut, nie 6h:
   ```python
   import faulthandler, signal, sys
   faulthandler.enable()
   faulthandler.register(signal.SIGUSR1, all_threads=True)
   faulthandler.dump_traceback_later(timeout=600, repeat=True, file=sys.stderr)
   ```
3. **`py-spy` z capability `cap_sys_ptrace=eip`** ustawioną raz przez admina daje natychmiastowy dump dowolnego procesu bez sudo — strongly recommended setup na maszynach treningowych.

## Co dalej

- [ ] Zabić proces 1948888 i jego dzieci.
- [ ] Zaaplikować fix w `utils/helpers.py:derangement`.
- [ ] Dodać faulthandler do `main.py`.
- [ ] Restart treningu.
- [ ] (Opcjonalnie) dodać unit test na `derangement` z patologicznymi inputami (`["A","A"]`, `["","","X"]`, `["A","A","B"]`).
