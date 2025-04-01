"""
main.py
"""


from utils.handlers.get         import *
from utils.handlers.populate    import *
from utils.handlers.update      import *
from utils.handlers.bloomberg   import *

from utils.messages             import *

from database.db_engine          import DB_Engine


if __name__ == '__main__':
    
    db = DB_Engine()
    db.connect()

    exit = False
    while not(exit):
        
        cmd = str(input(FIRST_MAIN_INPUT))
        
        if not(cmd.isdigit()):
            print(DIGIT_ONLY_WARNING)
        
        else:
            cmd = int(cmd)
            
            if cmd == 0:
                handle_get_command(db)
            elif cmd == 1:
                handle_populate_command(db)
            elif cmd == 2:
                handle_update_command(db)
            elif cmd == 3:
                handle_bloomberg_command(db)
            else:
                exit = True
        
        
