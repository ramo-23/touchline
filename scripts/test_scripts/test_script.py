"""
Just a quick little script to test if statsbomb actually receives data
"""

import statsbombpy as sb_check
from statsbombpy import sb

def main():
    print("Fetching competitions list...")
    competitions = sb.competitions()
    print(f"Success — found {len(competitions)} competition/season entries.")
    print(competitions.head())
 
if __name__ == "__main__":
    main()