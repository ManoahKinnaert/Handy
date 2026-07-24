from experimental import cad_test, hand_cad, track

def choose_experimental():
    cmd = input("If you wish to run track.py enter 1, for cad_test.py enter 2 and for hand_cad.py enter 3: ")
    if cmd == "1": track.run_hand_tracking_on_webcam()
    elif cmd == "2": cad_test.main()
    elif cmd == "3": hand_cad.main()
    elif cmd.lower() == "q": main()
    else: print("Invalid choice!")
    choose_experimental()

def choose_core_test_suite():
    pass 

def main():
    print("Welcome to the test runner!")
    cmd = input("If you wish to run an experiment enter 1 if you want to run the core test suite (to be added soon, maybe...) enter 2 (enter q to quit): ")
    if cmd == "1": choose_experimental()
    elif cmd == "2": choose_core_test_suite()
    elif cmd.lower() == "q": exit(0)
    else: print("Invalid choice!")
    main()

if __name__ == "__main__":
    main()