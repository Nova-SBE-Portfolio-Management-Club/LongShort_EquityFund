"""
Functions to handle the GET command
"""

from ..messages import GET_COMMAND_INPUT, DIGIT_ONLY_WARNING


from database import DB_Engine


def handle_get_country(db: DB_Engine):
    """Handle the GET country command."""
    option = input("Count (0)\n Show All (1)\n ")
    
    countries = db.get_all_countries()
    if option == "1":
        #Show all countries.
        for country in countries:
            print(country)
        
    elif option == "0":
        print(f'Total number of countries: {len(countries)}\n')
        
        
        
def handle_get_company(db: DB_Engine):
    """Handle the GET company command."""   
    option = input("Count (0)\nCount By Country (1)\nCount By Sector (2)\n")
    
    print()
    if option == "0":
        print(f"Total companies: {len(db.get_all_companies())}")

    elif option == "1":
        # Group By Country
        dix = db.count_companies_by_attr('country')
        for country in dix:
            print(f'{country:<15} {dix[country]:>5} companies')
        print()
            
    elif option == "2":
        # Group By Sector
        dix = db.count_companies_by_attr('sector')
        for country in dix:
            print(f'{country:<15} {dix[country]:>5} companies')
        print()
    
    
def handle_get_pairs(db: DB_Engine):
    """Handle an unfinished menu option."""
    print("Pair queries are not implemented in the menu yet.")


def handle_get_pricedata(db: DB_Engine):
    """Handle an unfinished menu option."""
    print("Price-data menu queries are not implemented in the menu yet.")


def handle_get_industry(db: DB_Engine):
    """Handle an unfinished menu option."""
    print("Industry queries are not implemented in the menu yet.")


def handle_get_command(db: DB_Engine):
    
    exit = False
    while not(exit):
        
        cmd = str(input( GET_COMMAND_INPUT ))

        
        if not(cmd.isdigit()):
            print(DIGIT_ONLY_WARNING)
        
        else:
            cmd = int(cmd)
            
            if cmd == 0:
                handle_get_country(db)
            elif cmd == 1:
                handle_get_company(db)
            elif cmd == 2:
                handle_get_pairs(db)
            elif cmd == 3:
                handle_get_pricedata(db)
            elif cmd == 4:
                handle_get_industry(db)
            else:
                exit = True