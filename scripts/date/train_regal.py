from program_refactoring.refactor_db import main_call

'''
main_call(
    collection_path="python_data/date/gpt-3.5-turbo_NL+SL/my_vectordb/",
    filter_every=5,
    refactor_every=5,
    task="python",
    dataset="date",
    tree_type="big_tree",
    max_tuple_size=3,
    do_retry=True,
    helpers_second=True
)
'''

from subprocess import Popen

proc = Popen(
    args=[
        'python', "program_refactoring/refactor_db.py",
        '--collection_path', 'python_data/date/gpt-3.5-turbo_NL+SL/my_vectordb/',
        '--filter_every', '5',
        '--refactor_every', '5',
        '--task', 'python',
        '--model_name', 'codellama/CodeLlama-7b-Instruct-hf',#'meta-llama/Meta-Llama-3.1-8B', #codellama/CodeLlama-7b-Instruct-hf',
        '--dataset', 'date',
        '--tree_type', 'big_tree',
        '--max_tuple_size', '3',
        '--do_retry',
        '--helpers_second'
    ],
)
proc.wait()