use bitvec::prelude::*;
use ahash::{AHashMap, AHashSet};

type Dimensions = (u8, u8);
#[derive(Debug, Clone, Copy)]
enum Move {
    Up,
    Down,
    Left,
    Right,
}
impl Move {
    fn vertical(&self) -> bool {
        use Move::*;
        match self {
            Up | Down => true,
            Left | Right => false,
        }
    }
}

const BOARD_WORDS: usize = 1;
const BOARD_ELEMS: usize = BOARD_WORDS * 64;
#[derive(Clone, Eq, PartialEq, Hash)]
struct Board {
    dirs: BitArray<Lsb0, [usize; BOARD_WORDS]>,
    r: u8,
    c: u8,
}
impl Board {
    fn print(&self, dims: Dimensions) {
        debug_assert!(self.r < dims.0 || self.r == dims.0 && self.c == 0);
        debug_assert!(self.c < dims.1);
        debug_assert!(self.dirs.len() >= (dims.0 * dims.1) as usize);
        // Convention: Under current location is false.
        debug_assert!(!self.dirs[(self.r * dims.1 + self.c) as usize]);
        for r in 0..dims.0 {
            for c in 0..dims.1 {
                let string = if r == self.r && c == self.c {
                    "o"
                } else {
                    let index = r * dims.1 + c;
                    if self.dirs[index as usize] {
                        "|"
                    } else {
                        "-"
                    }
                };
                print!("{}", string);
            }
            println!()
        }
    }
    // Returns whether move was successful
    fn make_move(&mut self, movement: Move, dims: Dimensions) -> bool {
        debug_assert!(self.r < dims.0);
        debug_assert!(self.c < dims.1);
        debug_assert!(self.dirs.len() >= (dims.0 * dims.1) as usize);
        let (new_r, new_c) = match movement {
            Move::Up => {
                if self.r == 0 {
                    return false;
                }
                (self.r - 1, self.c)
            }
            Move::Down => {
                if self.r == dims.0 - 1 {
                    return false;
                }
                (self.r + 1, self.c)
            }
            Move::Left => {
                if self.c == 0 {
                    return false;
                }
                (self.r, self.c - 1)
            }
            Move::Right => {
                if self.c == dims.1 - 1 {
                    return false;
                }
                (self.r, self.c + 1)
            }
        };
        let new_index = new_r * dims.1 + new_c;
        let new_bit = self.dirs[new_index as usize];
        if new_bit == movement.vertical() {
            self.dirs.set(new_index as usize, false);
            let old_index = self.r * dims.1 + self.c;
            self.dirs.set(old_index as usize, new_bit);
            self.r = new_r;
            self.c = new_c;
            true
        } else {
            false
        }
    }
    // row and col counts are remaining verts in row and col.
    // Sums must be equal.
    // Can speed up by fast-filling cols
    fn generate(
        self,
        row_counts: &[u8],
        col_counts: &[u8],
        cur_r: u8,
        cur_c: u8,
        dims: Dimensions,
    ) -> Vec<Board> {
        let debug = false;
        if debug {
            self.print(dims);
            println!("{:?} {:?} {} {}", row_counts, col_counts, cur_r, cur_c);
            println!();
        }
        debug_assert!(cur_r < dims.0 || cur_r == dims.0 && cur_c == 0);
        debug_assert!(cur_c < dims.1);
        debug_assert_eq!(row_counts.iter().sum::<u8>(), col_counts.iter().sum::<u8>());
        let count: u8 = row_counts.iter().sum();
        if count == 0 {
            return vec![self];
        }
        if cur_r == dims.0 {
            return vec![];
        }
        let (next_r, next_c) = if cur_c == dims.1 - 1 {
            (cur_r + 1, 0)
        } else {
            (cur_r, cur_c + 1)
        };
        let set_true = if row_counts[cur_r as usize] > 0
            && col_counts[cur_c as usize] > 0
            // cur_c must be greater than self.c, because both cursor and left are auto horiz
            && (cur_r != self.r || cur_c > self.c )
        {
            let mut new_board = self.clone();
            new_board.dirs.set((cur_r * dims.1 + cur_c) as usize, true);
            let mut new_row_counts = row_counts.to_owned();
            new_row_counts[cur_r as usize] -= 1;
            let mut new_col_counts = col_counts.to_owned();
            new_col_counts[cur_c as usize] -= 1;
            new_board.generate(&new_row_counts, &new_col_counts, next_r, next_c, dims)
        } else {
            vec![]
        };
        let mut set_false = if row_counts[cur_r as usize] < dims.1 - cur_c
            && col_counts[cur_c as usize] < dims.0 - cur_r
            && next_r < dims.0
        {
            self.generate(row_counts, col_counts, next_r, next_c, dims)
        } else {
            vec![]
        };
        set_false.extend(set_true);
        set_false
    }
}

// fills seen
fn component(board: &Board, dims: Dimensions, seen: &mut AHashMap<Board, Vec<Board>>) {
    let mut in_boards = vec![board.clone()];
    seen.insert(board.clone(), vec![]);
    loop {
        let mut out_boards = vec![];
        for board in &in_boards {
            for movement in [Move::Up, Move::Down, Move::Left, Move::Right] {
                let mut new_board = board.clone();
                new_board.make_move(movement, dims);
                let board_neighbors = seen.get_mut(board).expect("Present");
                board_neighbors.push(new_board.clone());
                if !seen.contains_key(&new_board) {
                    out_boards.push(new_board.clone());
                    seen.insert(new_board, vec![board.clone()]);
                } else {
                    let new_neighbors = seen.get_mut(&new_board).expect("Just checked");
                    new_neighbors.push(board.clone());
                }
            }
        }
        if out_boards.is_empty() {
            return;
        } else {
            in_boards = out_boards;
        }
    }
}

fn dijkstra(comp: &AHashMap<Board, Vec<Board>>, start_row: u8, dims: Dimensions) -> (usize, Board) {
    let mut cur_dist: Vec<&Board> = comp
        .keys()
        .filter(|board| {
            let index = start_row * dims.1;
            let bit = board.dirs[index as usize];
            !bit && !(start_row == board.r && 0 == board.c)
        })
        .collect();
    let mut seen: AHashSet<&Board> = cur_dist.iter().copied().collect();
    let mut steps = 0;
    loop {
        let mut next_dist = vec![];
        for board in &cur_dist {
            for neighbor in &comp[board] {
                if !seen.contains(&neighbor) {
                    seen.insert(neighbor);
                    next_dist.push(neighbor);
                }
            }
        }
        if next_dist.is_empty() {
            return (steps, cur_dist[0].clone());
        }
        cur_dist = next_dist;
        steps += 1;
    }
}

// All boards with this many verts in the rows and cols.
// Cursor counts as horiz for row and vert for col, so we subtract one from col.
// Must also be in starting position, c == 1.
fn generate(row_counts: &[u8], col_counts: &[u8], start_row: u8, dims: Dimensions) -> Vec<Board> {
    let debug = false;
    if debug {
        println!("{:?} {:?} {}", row_counts, col_counts, start_row);
    }
    if col_counts[1] == 0 {
        return vec![];
    }
    let mut fixed_col_counts = col_counts.to_owned();
    fixed_col_counts[1] -= 1;
    let row_sum: u8 = row_counts.iter().sum();
    let col_sum: u8 = fixed_col_counts.iter().sum();
    assert_eq!(row_sum, col_sum);

    let board = Board {
        dirs: BitArray::zeroed(),
        r: start_row,
        c: 1,
    };
    board.generate(row_counts, &fixed_col_counts, 0, 0, dims)
}

fn product(bound: u8, reps: u8) -> Vec<Vec<u8>> {
    if reps == 0 {
        vec![vec![]]
    } else {
        let mut out = vec![];
        for parital in product(bound, reps - 1) {
            for b in 0..=bound {
                let mut new = parital.clone();
                new.push(b);
                out.push(new);
            }
        }
        out
    }
}

fn search(dims: Dimensions) {
    let incremental_printing = true;
    assert!((dims.0 * dims.1) as usize <= BOARD_ELEMS);
    let mut row_counts_lists = vec![vec![]; (dims.0 * dims.1) as usize + 1];
    for row_counts in product(dims.1, dims.0) {
        let count: u8 = row_counts.iter().sum();
        row_counts_lists[count as usize].push(row_counts);
    }
    let mut col_counts_lists = vec![vec![]; (dims.0 * dims.1) as usize + 1];
    for col_counts in product(dims.0, dims.1) {
        let count: u8 = col_counts.iter().sum();
        col_counts_lists[count as usize].push(col_counts);
    }
    let mut max_depth = 0;
    let mut deepest = None;
    let mut comp = AHashMap::new();
    for sum in 0..dims.0 * dims.1 {
        for row_counts in &row_counts_lists[sum as usize] {
            for col_counts in &col_counts_lists[sum as usize + 1] {
                // By symmetry, can ignore lower half of start rows
                for start_row in 0..dims.0 / 2 {
                    let boards = generate(row_counts, col_counts, start_row, dims);
                    let mut boards_set: AHashSet<Board> = boards.into_iter().collect();
                    while !boards_set.is_empty() {
                        let board = boards_set.iter().next().expect("Nonempty");
                        component(board, dims, &mut comp);
                        let (dist, farthest) = dijkstra(&comp, start_row, dims);
                        if dist > max_depth {
                            if incremental_printing {
                                println!(
                                    "{} {:?} {:?} {}",
                                    dist, row_counts, col_counts, start_row
                                );
                                farthest.print(dims);
                                println!();
                            }
                            max_depth = dist;
                            deepest = Some((board.clone(), farthest));
                        }
                        for board in comp.keys() {
                            if board.r == start_row
                                && board.c == 1
                                && !board.dirs[(start_row * dims.1) as usize]
                            {
                                boards_set.remove(board);
                            }
                        }
                        comp.clear();
                    }
                }
            }
        }
    }
    let (board, farthest) = deepest.expect("Found one");
    board.print(dims);
    println!("Farthest in {}", max_depth);
    farthest.print(dims);
}

fn search_all() {
    for sum in 8..=12 {
        let dim_pairs = if sum % 2 == 0 {
            vec![(sum / 2, sum / 2)]
        } else {
            vec![(sum / 2, sum / 2 + 1), (sum / 2 + 1, sum / 2)]
        };
        for dims in dim_pairs {
            search(dims);
        }
    }
}

fn search_one() {
    let dimensions = (4, 4);
    assert!((dimensions.0 * dimensions.1) as usize <= BOARD_ELEMS);
    let board = Board {
        dirs: BitArray::new([0b1010101111000001]),
        r: 1,
        c: 1,
    };
    let mut comp = AHashMap::new();
    component(&board, dimensions, &mut comp);
    println!("Seen {}", comp.len());
    let target_board = Board {
        dirs: BitArray::new([0b0011101011011000]),
        r: 2,
        c: 2,
    };
    assert!(comp.contains_key(&target_board));
    let (dist, farthest) = dijkstra(&comp, 1, dimensions);
    println!("{}", dist);
    board.print(dimensions);
    println!();
    farthest.print(dimensions);
}

fn explore() {
    let mut board = Board {
        dirs: BitArray::new([0b0110100110010100]),
        r: 0,
        c: 1,
    };
    let dimensions = (4, 4);
    board.print(dimensions);
    for movement in [
        Move::Left,
        Move::Down,
        Move::Right,
        Move::Right,
        Move::Up,
        Move::Right,
        Move::Down,
    ] {
        println!();
        board.make_move(movement, dimensions);
        board.print(dimensions);
    }
}

fn main() {
    let choice = std::env::args()
        .nth(1)
        .map_or(2, |string| string.parse().expect("Number"));
    match choice {
        0 => explore(),
        1 => search_one(),
        2 => search((4, 4)),
        3 => search_all(),
        _ => unimplemented!(),
    }
}
