## Algorithme

Programé par [[deepmind]]

Les premières versions d'AlphaGo utilisent la [[méthode de Monte-Carlo]], guidée par un « _value network_ » et un « _policy network_ » (un _réseau de valeur_ et un _réseau d'objectifs_), tous deux implémentés en utilisant un [[réseau de neurones profond]].

AlphaGo a initialement été entraîné pour « imiter » les joueurs humains, en retrouvant les coups enregistrés lors de dizaines de milliers de parties menées par des joueurs experts. Une fois un certain niveau atteint, il s'est entraîné à jouer des millions de parties contre d'autres instances de lui-même, utilisant l'[[apprentissage par renforcement]] pour s'améliorer.

Version plus poussée : [[AlphaGo Zero]], une architecture simplifiée et n’utilisant plus ni la méthode de Monte-Carlo, ni des connaissances humaines, mais parvenant à des performances supérieures.