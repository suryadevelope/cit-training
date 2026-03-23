import tkinter as tk

# ================== GRID SETTINGS ==================
ROWS = 5
COLS = 5
CELL_SIZE = 80

# ================== ENVIRONMENT ==================
start = (0, 0)
goal = (4, 4)
obstacles = [(1,1), (2,2), (3,1)]

# ================== WINDOW ==================
root = tk.Tk()
root.title("🎮 Smart Grid World - AI Suggestion")

canvas = tk.Canvas(root, width=COLS*CELL_SIZE, height=ROWS*CELL_SIZE)
canvas.pack()

# ================== DRAW GRID ==================
def draw_grid(agent_pos):
    canvas.delete("all")

    for i in range(ROWS):
        for j in range(COLS):
            x1 = j * CELL_SIZE
            y1 = i * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            if (i, j) == agent_pos:
                color = "blue"
            elif (i, j) == goal:
                color = "green"
            elif (i, j) in obstacles:
                color = "red"
            else:
                color = "white"

            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

# ================== MOVE FUNCTION ==================
def move(state, direction):
    r, c = state

    if direction == "UP":
        r -= 1
    elif direction == "DOWN":
        r += 1
    elif direction == "LEFT":
        c -= 1
    elif direction == "RIGHT":
        c += 1

    # Boundary check
    if r < 0 or r >= ROWS or c < 0 or c >= COLS:
        return state

    # Obstacle check
    if (r, c) in obstacles:
        return state

    return (r, c)

# ================== BEST MOVE FUNCTION ==================
def get_best_move(state):
    r, c = state
    gr, gc = goal

    possible_moves = []

    # Priority towards goal
    if r < gr:
        possible_moves.append("DOWN")
    if c < gc:
        possible_moves.append("RIGHT")
    if r > gr:
        possible_moves.append("UP")
    if c > gc:
        possible_moves.append("LEFT")

    # Backup options
    possible_moves += ["UP", "DOWN", "LEFT", "RIGHT"]

    for move_dir in possible_moves:
        new_state = move(state, move_dir)
        if new_state != state:
            return move_dir

    return None

# ================== GAME STATE ==================
state = start
game_over = False

# ================== KEY CONTROL ==================
def key_press(event):
    global state, game_over

    if game_over:
        return

    key = event.keysym

    if key == "Up":
        direction = "UP"
    elif key == "Down":
        direction = "DOWN"
    elif key == "Left":
        direction = "LEFT"
    elif key == "Right":
        direction = "RIGHT"
    else:
        return

    new_state = move(state, direction)

    # 🚫 Blocked → Suggest best move
    if new_state == state:
        best = get_best_move(state)

        canvas.delete("hint")

        if best:
            canvas.create_text(
                COLS*CELL_SIZE//2,
                20,
                text=f"❌ Blocked! Try: {best}",
                font=("Arial", 14, "bold"),
                fill="orange",
                tags="hint"
            )
            print("Blocked! Suggested:", best)
        return

    # ✅ Valid move
    state = new_state
    draw_grid(state)

    # Remove old hint
    canvas.delete("hint")

    # 🎯 Goal check
    if state == goal:
        game_over = True
        canvas.create_text(
            COLS*CELL_SIZE//2,
            ROWS*CELL_SIZE//2,
            text="🎯 YOU WIN!",
            font=("Arial", 24, "bold"),
            fill="purple"
        )
        print("🎯 Goal Reached!")

# ================== RESET FUNCTION ==================
def reset_game():
    global state, game_over
    state = start
    game_over = False
    draw_grid(state)
    canvas.delete("hint")

# ================== BUTTON ==================
reset_btn = tk.Button(root, text="🔄 Restart", command=reset_game)
reset_btn.pack(pady=10)

# ================== KEY BIND ==================
root.bind("<Up>", key_press)
root.bind("<Down>", key_press)
root.bind("<Left>", key_press)
root.bind("<Right>", key_press)

# ================== START ==================
draw_grid(start)
root.mainloop()
