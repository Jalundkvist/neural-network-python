from ann import Ann
"""
TODO: Accuracy sometimes above 99%, sometimes 60%
TODO: Clean up
TODO: Create UI?
TODO: Update comments
TODO: test on RPI
TODO: Add RPI buttons & LED.
"""
EPOCHS = 30000
LEARN_RATE = 0.0255
DATA_FILE = "train_data"
OUTPUT_FILE = "output"

def main():
    ann = Ann(4,1,3,4)
    train_in, train_out = ann.read_file(DATA_FILE)
    ann.set_training_data(train_in, train_out)
    print("\033c", end="")
    print("Training initiating!\n\n")
    print(f"Number of epochs:\t{EPOCHS}")
    print(f"Learnrate:\t\t{LEARN_RATE}")
    ann.train(train_in, train_out, EPOCHS, LEARN_RATE)
    print(f"\ntraining done!\n")
    ann.write_file(EPOCHS, LEARN_RATE,OUTPUT_FILE)

if __name__ == "__main__":
    main()