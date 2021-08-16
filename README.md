Unit Rush Hour
----

[Unit Rush Hour](http://tromp.github.io/orimaze.html), or OriMaze, is a game
that's played on a rectangular grid filled with blocks that are constrained to move in one direction, vertical or horizontal.
There's one open square, which we can move a block into.

The goal of the game is to move a horizontal block to a designated square on the left edge of the board.
Empirically, this can be pretty hard - games can take a long time to complete.
The game that takes the longest on boards of size 1x1, 2x2, 3x3, ...
takes the following number of steps to win:

2x2 | 3x3 | 4x4 | 5x5 | 6x6
----|-----|-----|-----|-----
3   | 12  | 40  | 199 | 732
 
This data is from 2004, in the following paper: [Limits of Rush Hour Logic Complexity](http://tromp.github.io/rh.ps).
The goal of this program is to find the hardest 7x7 board, by brute-forcing farther.

The complexity of this problem is very much open. NL? PSPACE-complete? I have no idea, though I'd lean toward PSPACE-complete.
Best fit for the data so far is:

    0.185 * e^(1.38x)

which gives the following fit and predictions:

Kind       | 2x2 | 3x3 | 4x4 | 5x5 | 6x6 | 7x7  |  8x8
-----------|-----|-----|-----|-----|-----|------|------
Reality    |  3  | 12  | 40  | 199 | 732 |      |
Prediction |  3  | 12  | 46  | 184 | 729 | 2900 | 11529

Options:

    cargo run --release 0: A sequence of moves on an example board.
    cargo run --release 1: Discovering the hardest board that ends at a specific board
    cargo run --release 2: Discovering the hardest board for a bunch of small board sizes.
        Benchmark. About 5s on my laptop if nothing else is running.
    cargo run --release 3: Discover the hardest board for all sizes. Prints out a bunch of progress info.
    cargo run --release 4: Play the game.

I'm on an ongoing project to optimize the performance as much as I can,
both to solve the problem and to learn more about optimization.
