import os

def create_output_folder():
    os.makedirs("outputs", exist_ok=True)


def print_section(title):
    print("\n" + "="*40)
    print(title)
    print("="*40)