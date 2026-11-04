# acts as the bridge/translator between your engine logic (which works with FENs, board dicts, or strings) and PyTorch (which operates on multi-dimensional numerical tensors).

import re
import torch

TACTALARIS_PIECE_TO_CHANNEL = {
    # White Pieces (Channels 0 - 7)
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'C': 6, 'Z': 7,
    # Black Pieces (Channels 8 - 15)
    'p': 8, 'n': 9, 'b': 10, 'r': 11, 'q': 12, 'k': 13, 'c': 14, 'z': 15
}

def parse_ep_squares(ep_str: str):
    """
    Parses an en passant string which may contain:
    - '-' (no en passant)
    - Single square (e.g., 'f3')
    - Double square (e.g., 'f3f4')
    
    Returns a list of (row, col) tuples.
    """
    if ep_str == '-' or not ep_str:
        return []
    
    # Extract all file-rank pairs (a-j followed by 1-10) using regex
    # Matches patterns like 'f3', 'f10', 'a4', etc.
    pattern = r'([a-jA-J])(10|[1-9])'
    matches = re.findall(pattern, ep_str)
    
    coords = []
    for file_char, rank_str in matches:
        col = ord(file_char.lower()) - ord('a')
        rank = int(rank_str)
        # On a 10x10 board: Rank 10 is row 0, Rank 1 is row 9
        row = 10 - rank
        if 0 <= row < 10 and 0 <= col < 10:
            coords.append((row, col))
            
    return coords

def fen_to_tensor(fen_string: str) -> torch.Tensor:
    """
    Converts a Tactalaris FEN string into a (20, 10, 10) PyTorch Tensor,
    supporting single and double en passant target squares.
    """
    tensor = torch.zeros((20, 10, 10), dtype=torch.float32)
    
    parts = fen_string.strip().split()
    position_part = parts[0]
    active_color = parts[1]
    ep_square_str = parts[3] if len(parts) > 3 else '-'
    halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
    fullmove_number = int(parts[5]) if len(parts) > 5 else 1
    
    # 1. Parse Board Pieces (Channels 0-15)
    rows = position_part.split('/')
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                if char in TACTALARIS_PIECE_TO_CHANNEL:
                    tensor[TACTALARIS_PIECE_TO_CHANNEL[char], r, c] = 1.0
                c += 1
                
    # 2. Channel 16: Active Player (1.0 for White, 0.0 for Black)
    if active_color == 'w':
        tensor[16, :, :] = 1.0
        
    # 3. Channel 17: Half-Move Clock (Normalized by 100)
    tensor[17, :, :] = min(float(halfmove_clock) / 100.0, 1.0)
    
    # 4. Channel 18: Full-Move Number (Normalized by 100)
    tensor[18, :, :] = min(float(fullmove_number) / 100.0, 1.0)
    
    # 5. Channel 19: En Passant Target Squares (Single or Multi-square)
    ep_coords_list = parse_ep_squares(ep_square_str)
    for r, c in ep_coords_list:
        tensor[19, r, c] = 1.0
            
    return tensor
    
def square_to_index(square_str: str) -> int:
    """
    Converts an algebraic square string (e.g., 'j2') to a flat index 0-99.
    Assumes standard chess file letters 'a'-'j' and rank numbers '1'-'10'.
    """
    if not square_str or len(square_str) < 2:
        raise ValueError(f"Invalid square string: {square_str}")
        
    file_char = square_str[0]
    rank_str = square_str[1:]
    
    col = ord(file_char.lower()) - ord('a')
    rank = int(rank_str)
    
    # Rank 10 is at the top (row 0), Rank 1 is at the bottom (row 9)
    row = 10 - rank
    
    if not (0 <= row < 10 and 0 <= col < 10):
        raise ValueError(f"Square out of bounds for 10x10 board: {square_str}")
        
    return row * 10 + col

def move_to_index(move_dict: dict) -> int:
    """
    Converts a move dictionary from rules.get_legal_moves() into a unique 
    integer index in the range [0, 9999].
    
    Example input:
    {"from": "j2", "piece": "P", "to": "j3", "fen": "..."}
    """
    from_sq = move_dict["from"]
    to_sq = move_dict["to"]
    
    from_idx = square_to_index(from_sq)  # 0 to 99
    to_idx = square_to_index(to_sq)      # 0 to 99
    
    # Unique composite index mapping (From-Square * 100 + To-Square)
    return (from_idx * 100) + to_idx

# --- Verification Test ---
if __name__ == "__main__":
    # Test FEN provided with double en passant squares: 'f3f4'
    fen = "9k/10/10/10/10/4pP4/6p3/10/10/K9 b - f3f4 0 1"
    t = state_to_tensor(fen)
    
    print("Tensor Shape:", t.shape)  # torch.Size([20, 10, 10])
    
    # Check Channel 19 (En Passant plane)
    ep_plane = t[19]
    print("Number of active EN Passant squares on grid:", ep_plane.sum().item())  # Should be 2.0
