# zrownowazona_siec_neuronowa_dino_run
## Opis ogólny

Aplikacja jest przeznaczona do szkolenia agentów przy użyciu algorytmu DQN (Deep Q-Network) w środowisku gry internetowej. Agentów szkoli się w wielu wątkach równolegle, z każdym wątkiem odpowiedzialnym za szkolenie jednego agenta.

## Struktura kodu źródłowego
### Główne elementy
CustomCallback - Klasa dziedzicząca po BaseCallback, odpowiedzialna za dostosowanie callbacków dla algorytmu DQN. Zapisuje najlepszy wynik i model w określonych interwałach.

WebGame - Klasa reprezentująca środowisko gry internetowej, dziedzicząca po klasie Env z biblioteki Gymnasium. Odpowiada za symulację gry, interakcję z grą, i dostarcza obserwacje agentowi.

train_model - Funkcja, w której szkoli się model DQN w określonym wątku. Używa callbacku CustomCallback do monitorowania postępów szkolenia.

visualize_saved_models - Funkcja do wizualizacji nauczonych modeli. Ładuje zapisane modele i generuje graficzną reprezentację struktury sieci neuronowej.

### Zmienne globalne
TRAIN_DIR - Katalog, w którym zapisywane są nauczone modele.
LOG_DIR - Katalog, w którym zapisywane są logi dotyczące najlepszych wyników.
NUM_THREADS - Liczba wątków równoległych do szkolenia agentów.
NUM_OF_STEPS - Liczba kroków, jakie każdy agent wykonuje podczas szkolenia.

### Klasy i funkcje
1. CustomCallback
Metody:
__init__(self, env, check_freq, save_path, thread_id, verbose=1) - Konstruktor inicjalizujący obiekt.
_init_callback(self) - Inicjalizuje callback.
_on_step(self) - Wywoływane po każdym kroku szkolenia. Aktualizuje najlepszy wynik, zapisuje model w określonych interwałach.
log_best_reward(self) - Loguje najlepszy wynik do pliku.

2. WebGame
Metody:
__init__(self, thread_id) - Konstruktor inicjalizujący obiekt środowiska gry.
step(self, action) - Wykonuje krok gry na podstawie akcji agenta.
reset(self, seed=None) - Resetuje środowisko do stanu początkowego.
render(self) - Funkcja do renderowania gry (pusta, ponieważ renderowanie nie jest obsługiwane).
close(self) - Zamyka środowisko.
get_observation(self) - Pobiera obserwację z gry.
get_done(self) - Sprawdza, czy gra została zakończona.

3. train_model
Argumenty:
model_id - Identyfikator modelu/wątku.
env - Środowisko gry dla danego wątku.
Opis:
Funkcja szkoli model DQN w określonym wątku, korzystając z callbacku CustomCallback.

4. visualize_saved_models
Argumenty:
directory - Katalog, w którym zapisane są modele.
Opis:
Funkcja wczytuje nauczone modele DQN i generuje graficzne reprezentacje ich struktury w formie plików PNG.

## Instrukcja uruchomienia
Przed uruchomieniem aplikacji, aktywuj wirtualne srodowisko. Dla aplikacji wykonujacej obliczenia na cpu
### dino-cpu/Scripts/activate
Dla aplikacjii wykonujacej obliczenia na gpu
### dino-gpu/Scripts/activate

Skonfiguruj zmienne globalne, takie jak TRAIN_DIR, LOG_DIR, NUM_THREADS i NUM_OF_STEPS.

Uruchom funkcję main().