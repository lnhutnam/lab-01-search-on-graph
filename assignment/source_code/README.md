### AI - Lab 01 - Search
**To run project**
1. Create virtual environment (if necessary)

	`virtualenv venv`
	`source venv/bin/activate`
1. Install package

	`pip install -r requirement.txt`
2. Run code

Argument from command line. To run: 

`python main.py input_file search_algorithm`

*search_algorithm* must be one of ['bfs', 'dfs', 'ucs', 'a_star',...]

For example:
	`python main.py input.txt bfs`

Run commad with example `input.txt`, you can change with another input text file in this project:

Breadth-First Search: `python main.py input.txt bfs`

Depth-First Search: `python main.py input.txt dfs`

Uniform-cost Search: `python main.py input.txt ucs`

A* Search: `python main.py input.txt a_star`

Greedy Best-First Search: `python main.py input.txt greedy_heuristic`

**Note**: to run sample code: `python main.py input.txt`
