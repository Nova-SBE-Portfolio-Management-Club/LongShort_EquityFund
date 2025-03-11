"""
main.py
"""


from utils.handlers.get         import *
from utils.handlers.populate    import *
from utils.handlers.update      import *
from utils.handlers.bloomberg   import *

from utils.messages             import *


if __name__ == '__main__':

    exit = False
    while not(exit):
        
        cmd = str(input(FIRST_MAIN_INPUT))
        
        if not(cmd.isdigit()):
            print(DIGIT_ONLY_WARNING)
        
        else:
            cmd = int(cmd)
            
            if cmd == 0:
                handle_get_command()
            elif cmd == 1:
                handle_populate_command()
            elif cmd == 2:
                handle_update_command()
            elif cmd == 3:
                handle_bloomberg_command()
            else:
                exit = True
        
        
