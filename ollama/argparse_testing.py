import argparse

parser = argparse.ArgumentParser()
parser.add_argument("text", help="input text")
parser.add_argument("-r", "--reason", action="store_true", default=False,
                    help="enable reasoning mode")
parser.add_argument("-m", "--model", default=None,
                    help="model name to use")
parser.add_argument("-i", "--interactive", action="store_true", default=False,
                    help="run in interactive mode")

args = parser.parse_args()

print("TEXT: ",args.text)         # "input text bla bla bla"
print("REASON: ",args.reason)       # True if -r/--reason given, else False
print("MODEL: ",args.model)        # model name string, or None
print("INTERACTIVE", args.interactive)  # True if -i/--interactive given, else False