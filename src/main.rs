use ahash::{AHashMap, AHashSet};

use std::collections::hash_map::Entry;
use std::time::SystemTime;

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
const MOVE_ARRAY: [Move; 4] = [Move::Up, Move::Down, Move::Left, Move::Right];

fn _bits_to_vec(num: u64, len: u8) -> Vec<bool> {
    assert!(len <= 64);
    (0..len).map(|i| num & (1 << i) != 0).collect()
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
struct Bits(u64);
impl Bits {
    fn is_set(&self, index: u8) -> bool {
        let inner = self.0;
        inner & 1 << index != 0
    }
    fn is_unset(&self, index: u8) -> bool {
        let inner = self.0;
        inner & 1 << index == 0
    }
    fn set(&mut self, index: u8) {
        self.0 |= 1 << index;
    }
    fn unset(&mut self, index: u8) {
        self.0 &= !(1 << index);
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
struct Board {
    dirs: Bits,
    r: u8,
    c: u8,
}
impl Board {
    fn print(&self, dims: Dimensions) {
        assert!(self.r < dims.0 || self.r == dims.0 && self.c == 0);
        assert!(self.c < dims.1);
        assert!(64 >= dims.0 * dims.1);
        // Convention: Under current location is false.
        assert!(self.dirs.is_unset(self.r * dims.1 + self.c));
        for r in 0..dims.0 {
            for c in 0..dims.1 {
                let string = if r == self.r && c == self.c {
                    "o"
                } else {
                    let index = r * dims.1 + c;
                    if self.dirs.is_set(index) {
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
        debug_assert!(64 >= dims.0 * dims.1);
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
        let new_bit = self.dirs.is_set(new_index);
        if new_bit == movement.vertical() {
            let old_index = self.r * dims.1 + self.c;
            // Set if new_bit. Was false
            if new_bit {
                self.dirs.set(old_index)
            }
            // Unset
            self.dirs.unset(new_index);
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
        mut self,
        row_counts: &[u8],
        col_counts: &[u8],
        cur: u8,
        total: u8,
        dims: Dimensions,
    ) -> Vec<Board> {
        debug_assert_eq!(row_counts.iter().sum::<u8>(), col_counts.iter().sum::<u8>());
        if total == 0 {
            return vec![self];
        }
        let cur_r = cur / dims.1;
        let cur_c = cur % dims.1;
        if cur_r >= dims.0 {
            return vec![];
        }
        if dims.0 * dims.1 - cur < total {
            return vec![];
        }
        if dims.0 * dims.1 - cur == total {
            for index in cur..dims.0 * dims.1 {
                self.dirs.set(index);
            }
            return vec![self];
        }
        // Blockage set in place. Note that we never have self.r == dims.0 - 1
        if cur == (self.r + 1) * dims.1 + 1 && self.prefilter_out(dims) {
            return vec![];
        }
        let set_true = if row_counts[cur_r as usize] > 0
            && col_counts[cur_c as usize] > 0
            // cur_c must be greater than self.c, because both cursor and left are auto horiz
            && (cur_r != self.r || cur_c > self.c )
        {
            let mut new_board = self;
            new_board.dirs.set(cur);
            let mut new_row_counts = row_counts.to_owned();
            new_row_counts[cur_r as usize] -= 1;
            let mut new_col_counts = col_counts.to_owned();
            new_col_counts[cur_c as usize] -= 1;
            new_board.generate(&new_row_counts, &new_col_counts, cur + 1, total - 1, dims)
        } else {
            vec![]
        };
        let mut set_false = if row_counts[cur_r as usize] < dims.1 - cur_c
            && col_counts[cur_c as usize] < dims.0 - cur_r
        {
            self.generate(row_counts, col_counts, cur + 1, total, dims)
        } else {
            vec![]
        };
        set_false.extend(set_true);
        set_false
    }
    // Can't possibly succeed.
    fn prefilter_out(&self, dims: Dimensions) -> bool {
        debug_assert!(self.c == 1);
        let up_blocked = self.r == 0 || self.dirs.is_unset((self.r - 1) * dims.1);
        let down_blocked = self.r == dims.0 - 1 || self.dirs.is_unset((self.r + 1) * dims.1);
        up_blocked && down_blocked
    }
}

// Holds seen, in_boards, out_boards to avoid reallocation
struct ComponentSearcher {
    seen: AHashMap<Board, ([bool; 4], bool)>,
    in_boards: Vec<(Board, usize)>,
    out_boards: Vec<(Board, usize)>,
}
impl ComponentSearcher {
    fn new() -> Self {
        Self {
            seen: AHashMap::new(),
            in_boards: vec![],
            out_boards: vec![],
        }
    }
    // Returns reference to seen.
    fn component(
        &mut self,
        board: &Board,
        dims: Dimensions,
    ) -> &mut AHashMap<Board, ([bool; 4], bool)> {
        self.seen.clear();
        self.in_boards.clear();
        self.out_boards.clear();
        self.in_boards.push((*board, 2));
        self.seen.insert(*board, ([false; 4], false));
        assert!(board.c == 1);
        loop {
            for (search_board, came_from) in self.in_boards.drain(..) {
                let mut successes = vec![];
                for (i, &movement) in MOVE_ARRAY.iter().enumerate() {
                    if &search_board == board {
                        if i != 2 {
                            continue;
                        }
                    } else if i == came_from {
                        continue;
                    }
                    let mut new_board = search_board;
                    let movable = new_board.make_move(movement, dims);
                    if movable {
                        let reverse_i = i ^ 1;
                        let entry = self.seen.entry(new_board);
                        if let Entry::Vacant(_) = entry {
                            self.out_boards.push((new_board, reverse_i));
                        }
                        let entry = entry.or_insert(([false; 4], false));
                        entry.0[reverse_i as usize] = true;
                        successes.push(i);
                    }
                }
                if !successes.is_empty() {
                    let board_neighbors = &mut self.seen.get_mut(&search_board).expect("Present").0;
                    for i in successes {
                        board_neighbors[i as usize] = true;
                    }
                }
            }
            if self.out_boards.is_empty() {
                return &mut self.seen;
            } else {
                std::mem::swap(&mut self.in_boards, &mut self.out_boards);
            }
        }
    }
}

// Holds cur_dist, next_dist to avoid reallocation
struct DijkstraSearcher {
    cur_dist: Vec<(Board, [bool; 4], usize)>,
    next_dist: Vec<(Board, [bool; 4], usize)>,
}
impl DijkstraSearcher {
    fn new() -> Self {
        Self {
            cur_dist: vec![],
            next_dist: vec![],
        }
    }
    fn dijkstra(
        &mut self,
        map: &mut AHashMap<Board, ([bool; 4], bool)>,
        start_row: u8,
        dims: Dimensions,
    ) -> (usize, Board) {
        debug_assert!(map.values().all(|(_, seen)| !seen));
        self.cur_dist.clear();
        self.next_dist.clear();
        self.cur_dist.extend(
            map.iter()
                .filter(|(board, _)| {
                    start_row == board.r
                        && 0 == board.c
                        && board.dirs.is_unset(start_row * dims.1 + 1)
                })
                .map(|(board, (array, _))| (*board, *array, 3)),
        );
        debug_assert!(!self.cur_dist.is_empty());
        let mut steps = 1;
        loop {
            for &(board, array, come_from) in &self.cur_dist {
                debug_assert!(
                    board.dirs.is_set(start_row * dims.1) || start_row == board.r && 0 == board.c
                );
                for (i, &b) in array.iter().enumerate() {
                    if b && i != come_from {
                        let mut neighbor = board;
                        neighbor.make_move(MOVE_ARRAY[i], dims);
                        let (neighbor_array, ref mut seen) =
                            map.get_mut(&neighbor).expect("Present");
                        if !*seen {
                            let reverse_i = i ^ 1;
                            self.next_dist.push((neighbor, *neighbor_array, reverse_i));
                        }
                        *seen = true;
                    }
                }
            }
            if self.next_dist.is_empty() {
                return (steps, self.cur_dist[0].0);
            }
            self.cur_dist.clear();
            std::mem::swap(&mut self.cur_dist, &mut self.next_dist);
            steps += 1;
        }
    }
}

// All boards with this many verts in the rows and cols.
// Cursor counts as vert for row and horiz for col, so we subtract 1 from row.
// Must also be in starting position, c == 1.
fn generate(row_counts: &[u8], col_counts: &[u8], start_row: u8, dims: Dimensions) -> Vec<Board> {
    let debug = false;
    let mut all_outs = vec![];
    if debug {
        println!("{:?} {:?} {}", row_counts, col_counts, start_row);
    }
    if row_counts[start_row as usize] == 0 {
        return vec![];
    }
    let mut fixed_row_counts = row_counts.to_owned();
    fixed_row_counts[start_row as usize] -= 1;
    let row_sum: u8 = fixed_row_counts.iter().sum();
    let col_sum: u8 = col_counts.iter().sum();
    assert_eq!(row_sum, col_sum);
    if start_row * 2 + 1 == dims.0 {
        let rev_rows: Vec<u8> = row_counts.iter().cloned().rev().collect();
        if row_counts < &rev_rows {
            return vec![];
        }
    }

    let board = Board {
        dirs: Bits(0),
        r: start_row,
        c: 1,
    };
    let results = board.generate(&fixed_row_counts, col_counts, 0, row_sum, dims);
    all_outs.extend(results);
    all_outs
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

fn search(dims: Dimensions, incremental_printing: bool) {
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
    let frequency = 1_000_000;
    let mut comps_search = 0;
    let mut last_print_time = 0;
    let time_frequency = 60;
    let start = SystemTime::now();
    let mut component_searcher = ComponentSearcher::new();
    let mut dijkstra_searcher = DijkstraSearcher::new();
    for sum in 0..dims.0 * dims.1 {
        for row_counts in &row_counts_lists[sum as usize + 1] {
            // 0: inaccessible, Max: decisionless
            if row_counts.iter().any(|r| *r == 0 || *r == dims.1) {
                continue;
            }
            for col_counts in &col_counts_lists[sum as usize] {
                // 0: decisionless, Max: inaccessible
                if col_counts
                    .iter()
                    .enumerate()
                    .any(|(i, c)| *c == 0 || *c == dims.0 && i != 1)
                {
                    continue;
                }
                for start_row in 0..(dims.0 + 1) / 2 {
                    let boards = generate(row_counts, col_counts, start_row, dims);
                    let mut boards_set: AHashSet<Board> = boards.into_iter().collect();
                    while !boards_set.is_empty() {
                        let board = *boards_set.iter().next().expect("Nonempty");
                        let map = component_searcher.component(&board, dims);
                        map.retain(|&comp_board, _| {
                            if comp_board.dirs.is_set(start_row * dims.1) {
                                true
                            } else {
                                if comp_board.r == start_row && comp_board.c == 1 {
                                    boards_set.remove(&comp_board);
                                }
                                comp_board.r == start_row && comp_board.c == 0
                            }
                        });
                        let (dist, farthest) = dijkstra_searcher.dijkstra(map, start_row, dims);
                        if dist > max_depth {
                            if incremental_printing {
                                println!(
                                    "{} {} {:?} {:?} {} {} {} {}",
                                    dist,
                                    sum,
                                    row_counts,
                                    col_counts,
                                    start_row,
                                    map.len(),
                                    comps_search,
                                    start.elapsed().expect("Positive").as_secs()
                                );
                                farthest.print(dims);
                                println!();
                            }
                            max_depth = dist;
                            deepest = Some((start_row, farthest));
                        }
                        comps_search += 1;
                        if incremental_printing && comps_search % frequency == 0 {
                            let time = start.elapsed().expect("Positive").as_secs();
                            if time - last_print_time > time_frequency {
                                last_print_time = time;
                                println!(
                                    "{} {} {:?} {:?} {} {} {}",
                                    comps_search,
                                    sum,
                                    row_counts,
                                    col_counts,
                                    map.len(),
                                    boards_set.len(),
                                    time,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    let (start_row, farthest) = deepest.expect("Found one");
    println!(
        "Farthest in {}, time {}: Row {}",
        max_depth,
        start.elapsed().expect("Positive").as_secs(),
        start_row
    );
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
            search(dims, true);
        }
    }
}

fn search_one() {
    let dimensions = (4, 4);
    let board = Board {
        dirs: Bits(0b1010101111000001),
        r: 1,
        c: 1,
    };
    let mut component_searcher = ComponentSearcher::new();
    let map = component_searcher.component(&board, dimensions);
    println!("Seen {}", map.len());
    let target_board = Board {
        dirs: Bits(0b0011101011011000),
        r: 2,
        c: 2,
    };
    assert!(map.contains_key(&target_board));
    let mut dijkstra_searcher = DijkstraSearcher::new();
    let (dist, farthest) = dijkstra_searcher.dijkstra(map, 1, dimensions);
    println!("{}", dist);
    board.print(dimensions);
    println!();
    farthest.print(dimensions);
}

fn explore() {
    let mut board = Board {
        dirs: Bits(0b0110100110010100),
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

fn play(option: usize) {
    let (mut board, dims, target_row) = match option {
        0 => (
            Board {
                dirs: Bits(0b0011101011011000),
                r: 2,
                c: 2,
            },
            (4, 4),
            1,
        ),
        1 => (
            Board {
                dirs: Bits(0b00011001101101111110),
                r: 3,
                c: 2,
            },
            (5, 4),
            1,
        ),
        _ => unimplemented!(),
    };
    let mut steps = 0;
    println!("wasd keys");
    while board.dirs.is_set(target_row * dims.1) || board.r == target_row && board.c == 0 {
        println!("Steps {}, Goal row {}", steps, target_row);
        board.print(dims);
        let mut string = String::new();
        std::io::stdin().read_line(&mut string).expect("Input");
        let movement = match string.chars().next() {
            Some('w') => Move::Up,
            Some('a') => Move::Left,
            Some('s') => Move::Down,
            Some('d') => Move::Right,
            _ => continue,
        };
        let was_successful = board.make_move(movement, dims);
        if was_successful {
            steps += 1;
        }
    }
    println!("Completed in {} steps!", steps);
}

fn main() {
    let choice = std::env::args()
        .nth(1)
        .map_or(2, |string| string.parse().expect("Number"));
    match choice {
        0 => explore(),
        1 => search_one(),
        2 => {
            for dims in [(4, 4), (4, 5), (5, 4), (4, 6)] {
                search(dims, false);
            }
        }
        3 => search_all(),
        4 => play(1),
        _ => unimplemented!(),
    }
}
