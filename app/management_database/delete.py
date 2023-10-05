## IMPORT ##########

import os
import sqlite3

####################

def main() :

    MAIN_PATH = os.getenv("MINMAX_ODM_MAINPATH").replace("\r", "").replace("\n", "")
    conn = sqlite3.connect(MAIN_PATH+"/data/database/my_database.db")
    cursor = conn.cursor()
    cursor.execute("""DELETE FROM all_run""")
    cursor.execute("""DELETE FROM all_performances""")
    cursor.execute("""DELETE FROM all_features_usage""")
    cursor.execute("""DELETE FROM dtc_parameters""")
    cursor.execute("""DELETE FROM rfc_parameters""")
    conn.commit()

if __name__ == "__main__" :
    main()