import numpy as np
import random
from my2048 import Numpy2048
import time

class AI2048:
    def __init__(self, max_depth=4, max_time=0.15):
        self.max_depth = max_depth
        self.max_time = max_time
        self.start_time = 0
        self.moves = {0: 'up', 1: 'left', 2: 'down', 3: 'right'}
        self.preferred_move_order = [0, 1, 3, 2]  # up, left, right, down preference
        
        # Plusieurs matrices de poids pour différentes stratégies
        # Snake pattern classique (en zigzag)
        self.snake_pattern = np.array([
            [2**15, 2**14, 2**13, 2**12],
            [2**8,  2**9,  2**10, 2**11],
            [2**7,  2**6,  2**5,  2**4],
            [2**0,  2**1,  2**2,  2**3]
        ])
        
        # Pattern de coin (pour maximiser dans le coin supérieur gauche)
        self.corner_pattern = np.array([
            [2**16, 2**15, 2**14, 2**13],
            [2**9,  2**10, 2**11, 2**12],
            [2**8,  2**7,  2**6,  2**5],
            [2**1,  2**2,  2**3,  2**4]
        ])
        
        # Pattern de monotonie (décroissance stricte)
        self.monotonic_pattern = np.array([
            [16, 15, 14, 13],
            [9,  10, 11, 12],
            [8,  7,  6,  5],
            [1,  2,  3,  4]
        ])
    
    def evaluate_board(self, board):
        """Évaluation complète du plateau"""
        max_value = np.max(board)
        empty_cells = np.sum(board == 0)
        
        # 1. FACTEUR CLEF : Priorité aux cellules vides (facteur de survie)
        empty_score = empty_cells**2 * 100  # Valeur quadratique pour les cases vides
        
        # 2. STRUCTURATION : Essayer différents modèles et prendre le meilleur
        pattern_scores = []
        
        # Snake pattern
        snake_score = self._apply_pattern(board, self.snake_pattern)
        pattern_scores.append(snake_score)
        
        # Corner pattern
        corner_score = self._apply_pattern(board, self.corner_pattern)
        pattern_scores.append(corner_score)
        
        # Tester toutes les orientations des patterns
        flipped_board = np.fliplr(board)  # Retourné horizontalement 
        pattern_scores.append(self._apply_pattern(flipped_board, self.snake_pattern))
        pattern_scores.append(self._apply_pattern(flipped_board, self.corner_pattern))
        
        # Prendre le meilleur score parmi tous les patterns
        structure_score = max(pattern_scores) * 2.0
        
        # 3. MONOTONIE : Calculer le degré de monotonie (crucial pour 2048)
        monotonicity = self._calculate_advanced_monotonicity(board)
        
        # 4. POTENTIEL DE FUSION : Encourager les tuiles fusionnables
        merge_score = self._calculate_merge_potential(board) * 10
        
        # 5. LISSAGE : Favoriser les tuiles de valeurs proches
        smoothness = self._calculate_smoothness(board) * 5
        
        # 6. BONUS DE COIN : Favoriser les grandes valeurs dans les coins
        corner_bonus = 0
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        corner_bonus = max(corners) * 5
        
        # 7. BONUS DE GRAND NOMBRE : Encourager la création de grandes tuiles
        value_score = np.sum(board) * 0.5 + (max_value**2) * 0.1
        
        # 8. PÉNALITÉ DE BLOCAGE : Éviter le blocage des grandes tuiles au milieu
        blocking_penalty = self._calculate_blocking_penalty(board)
        
        # Combinaison pondérée des facteurs (calibré empiriquement)
        final_score = (
            empty_score * 100 +        # Facteur dominant - cases vides
            structure_score * 30 +      # Structure (snake/corner) 
            monotonicity * 20 +        # Monotonie
            merge_score * 10 +         # Fusions possibles
            smoothness * 5 +           # Lissage
            corner_bonus * 4 +         # Bonus de coin
            value_score +              # Score brut
            blocking_penalty * -50     # Pénalité de blocage
        )
        
        return final_score
    
    def _apply_pattern(self, board, pattern):
        """Applique un pattern au plateau et retourne le score"""
        return np.sum(board * pattern)
    
    def _calculate_advanced_monotonicity(self, board):
        """Calcul amélioré de la monotonie"""
        # Convertir les valeurs pour mieux gérer les ratios
        values = np.zeros_like(board, dtype=float)
        mask = board > 0
        values[mask] = np.log2(board[mask])
        
        total_monotonicity = 0
        
        # Pour chaque ligne et chaque colonne, calculer dans les deux directions
        for i in range(4):
            # Monotonie horizontale
            row = values[i]
            left_sum = 0
            right_sum = 0
            
            for j in range(3):
                if row[j] >= row[j+1]:  # Décroissant de gauche à droite
                    left_sum += (row[j] - row[j+1])
                else:
                    left_sum -= (row[j+1] - row[j]) * 2  # Pénalité plus forte
                
                if row[3-j] >= row[3-j-1]:  # Décroissant de droite à gauche
                    right_sum += (row[3-j] - row[3-j-1])
                else:
                    right_sum -= (row[3-j-1] - row[3-j]) * 2
            
            # Prendre le meilleur des deux
            total_monotonicity += max(left_sum, right_sum)
            
            # Monotonie verticale
            col = values[:, i]
            up_sum = 0
            down_sum = 0
            
            for j in range(3):
                if col[j] >= col[j+1]:  # Décroissant de haut en bas
                    up_sum += (col[j] - col[j+1])
                else:
                    up_sum -= (col[j+1] - col[j]) * 2
                
                if col[3-j] >= col[3-j-1]:  # Décroissant de bas en haut
                    down_sum += (col[3-j] - col[3-j-1])
                else:
                    down_sum -= (col[3-j-1] - col[3-j]) * 2
            
            total_monotonicity += max(up_sum, down_sum)
        
        return total_monotonicity
    
    def _calculate_smoothness(self, board):
        """Calcule le lissage entre tuiles adjacentes (valeurs proches = bon)"""
        smoothness = 0
        log_board = np.zeros_like(board, dtype=float)
        mask = board > 0
        log_board[mask] = np.log2(board[mask])
        
        for i in range(4):
            for j in range(4):
                if board[i, j] > 0:
                    # Différence avec la droite
                    if j < 3 and board[i, j+1] > 0:
                        smoothness -= abs(log_board[i, j] - log_board[i, j+1])
                    # Différence avec le bas
                    if i < 3 and board[i+1, j] > 0:
                        smoothness -= abs(log_board[i, j] - log_board[i+1, j])
        
        return smoothness
    
    def _calculate_merge_potential(self, board):
        """Calcule le potentiel de fusion des tuiles adjacentes"""
        merge_score = 0
        
        # Vérifier les fusions horizontales avec valeur logarithmique
        for i in range(4):
            for j in range(3):
                if board[i, j] > 0 and board[i, j] == board[i, j+1]:
                    merge_score += board[i, j] * 2  # Récompense plus élevée pour les grandes valeurs
        
        # Vérifier les fusions verticales
        for i in range(3):
            for j in range(4):
                if board[i, j] > 0 and board[i, j] == board[i+1, j]:
                    merge_score += board[i, j] * 2
        
        return merge_score
    
    def _calculate_blocking_penalty(self, board):
        """Pénalise le blocage des grandes tuiles au milieu du plateau"""
        penalty = 0
        max_tile = np.max(board)
        
        # Si grande tuile (>= 256) est entourée de petites tuiles, c'est mauvais
        for i in range(1, 3):
            for j in range(1, 3):
                if board[i, j] >= 256:
                    # Vérifier si elle est bloquée
                    is_blocked = True
                    surrounding_cells = [
                        board[i-1, j], board[i+1, j], 
                        board[i, j-1], board[i, j+1]
                    ]
                    
                    # Si entourée de tuiles beaucoup plus petites, c'est un blocage
                    for cell in surrounding_cells:
                        if cell == 0 or cell >= board[i, j] / 2:
                            is_blocked = False
                            break
                    
                    if is_blocked:
                        penalty += board[i, j]
        
        return penalty
    
    def expectimax(self, game, depth, maximize=True):
        """Algorithme expectimax optimisé"""
        # Vérifier si le temps est écoulé
        if time.time() - self.start_time > self.max_time:
            return self.evaluate_board(game.board), None
        
        # Cas de base
        if depth == 0 or game.is_game_over():
            return self.evaluate_board(game.board), None
        
        if maximize:
            # Tour du joueur (maximiser le score)
            max_score = float('-inf')
            best_move = None
            
            # Tester les mouvements dans un ordre préférentiel
            for move_idx in self.preferred_move_order:
                move_dir = self.moves[move_idx]
                
                # Copier et simuler le mouvement
                game_copy = Numpy2048(game.board.shape[0])
                game_copy.board = game.board.copy()
                
                # Si le mouvement est valide
                if game_copy.move(move_dir):
                    # Appel récursif
                    eval_score, _ = self.expectimax(game_copy, depth-1, False)
                    
                    # Mettre à jour le meilleur score
                    if eval_score > max_score:
                        max_score = eval_score
                        best_move = move_idx
            
            # Si aucun mouvement valide
            if best_move is None:
                return float('-inf'), None
            
            return max_score, best_move
            
        else:
            # Tour de l'ordinateur (placement aléatoire de tuile)
            avg_score = 0
            total_weight = 0
            
            # Trouver les cellules vides
            empty_cells = np.where(game.board == 0)
            empty_positions = list(zip(empty_cells[0], empty_cells[1]))
            
            if not empty_positions:
                return self.evaluate_board(game.board), None
            
            # Si trop de cellules vides, échantillonner intelligemment
            if len(empty_positions) > 5:
                # Sélectionner les positions les plus stratégiques
                empty_positions = self._select_strategic_positions(game.board, empty_positions, 4)
            
            # Pour chaque position vide, simuler l'ajout d'une tuile
            for pos in empty_positions:
                # Simuler l'ajout d'une tuile 2 (90% de probabilité)
                game_copy = Numpy2048(game.board.shape[0])
                game_copy.board = game.board.copy()
                game_copy.board[pos] = 2
                
                # Score avec une tuile 2
                score_with_2, _ = self.expectimax(game_copy, depth-1, True)
                
                # Simuler l'ajout d'une tuile 4 (10% de probabilité)
                game_copy.board[pos] = 4
                score_with_4, _ = self.expectimax(game_copy, depth-1, True)
                
                # Calculer le score moyen pondéré
                weighted_score = (score_with_2 * 0.9 + score_with_4 * 0.1)
                avg_score += weighted_score
                total_weight += 1
            
            # Retourner le score moyen
            if total_weight > 0:
                return avg_score / total_weight, None
            else:
                return self.evaluate_board(game.board), None
    
    def _select_strategic_positions(self, board, positions, num_select):
        """Sélectionne les positions les plus stratégiques pour l'échantillonnage"""
        # Stratégie : préférer les positions qui sont adjacentes à des tuiles existantes
        # et qui ne sont pas dans les coins si possible
        
        # Calculer un score pour chaque position
        position_scores = []
        
        for pos in positions:
            i, j = pos
            score = 0
            
            # Vérifier les voisins (combien ont des tuiles non nulles)
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for ni, nj in neighbors:
                if 0 <= ni < 4 and 0 <= nj < 4 and board[ni, nj] > 0:
                    score += 1
            
            # Bonus pour les positions centrales
            if 1 <= i <= 2 and 1 <= j <= 2:
                score += 0.5
            
            position_scores.append((pos, score))
        
        # Trier par score décroissant
        position_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les meilleures positions
        return [pos for pos, _ in position_scores[:num_select]]
    
    def get_best_move(self, board):
        """Détermine le meilleur mouvement pour un plateau donné"""
        # Réinitialiser le temps
        self.start_time = time.time()
        
        # Créer un objet jeu
        game = Numpy2048(board.shape[0])
        game.board = board.copy()
        
        # Calculer la profondeur optimale
        empty_count = np.sum(board == 0)
        if empty_count <= 2:
            depth = min(5, self.max_depth + 1)  # Situation critique - recherche plus profonde
        elif empty_count <= 4:
            depth = min(4, self.max_depth)
        else:
            depth = self.max_depth
        
        # Exécuter l'algorithme expectimax
        _, best_move = self.expectimax(game, depth, True)
        
        # Si aucun mouvement valide n'est trouvé
        if best_move is None:
            valid_moves = []
            for move_idx, move_dir in self.moves.items():
                game_copy = Numpy2048(board.shape[0])
                game_copy.board = board.copy()
                if game_copy.move(move_dir):
                    valid_moves.append(move_idx)
            
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return random.randint(0, 3)
        
        return best_move

# Fonction compatible avec l'interface existante
def best_move(board, depth=3):
    """Fonction wrapper pour maintenir la compatibilité"""
    ai = AI2048(max_depth=depth)
    return ai.get_best_move(board)
