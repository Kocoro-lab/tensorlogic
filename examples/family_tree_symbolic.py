"""
Family Tree Reasoning Example - Symbolic Mode

Demonstrates logical reasoning using Boolean tensors.
"""

import torch
from tensorlogic import TensorProgram
from tensorlogic.reasoning.forward import forward_chain


def main():
    print("=" * 60)
    print("Family Tree Reasoning - Symbolic Mode")
    print("=" * 60)

    # Create a program in Boolean mode
    program = TensorProgram(mode='boolean', device='cpu')

    # Define family members (indices)
    # 0: Alice, 1: Bob, 2: Charlie, 3: Diana, 4: Eve
    num_people = 5
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

    print(f"\nFamily members: {names}")

    # Define parent relation as facts
    # Alice is parent of Bob and Charlie
    # Bob is parent of Diana
    # Charlie is parent of Eve
    parent_data = torch.zeros(num_people, num_people)
    parent_data[0, 1] = 1  # Alice -> Bob
    parent_data[0, 2] = 1  # Alice -> Charlie
    parent_data[1, 3] = 1  # Bob -> Diana
    parent_data[2, 4] = 1  # Charlie -> Eve

    print("\nParent relationships:")
    for i in range(num_people):
        for j in range(num_people):
            if parent_data[i, j] == 1:
                print(f"  {names[i]} is parent of {names[j]}")

    # Add parent facts to program
    program.add_tensor('parent', data=parent_data, learnable=False)

    # Add rule: grandparent(X, Z) <- parent(X, Y), parent(Y, Z)
    # In tensor form: grandparent = parent @ parent
    program.add_equation('grandparent', 'parent @ parent')

    # Execute forward chaining
    print("\n" + "=" * 60)
    print("Executing forward chaining...")
    print("=" * 60)

    results = forward_chain(program, {'parent': parent_data}, max_iterations=10)

    # Display grandparent relationships
    print("\nGrandparent relationships (derived):")
    grandparent = results['grandparent']
    for i in range(num_people):
        for j in range(num_people):
            if grandparent[i, j] > 0.5:
                print(f"  {names[i]} is grandparent of {names[j]}")

    # Query specific relationship
    print("\n" + "=" * 60)
    print("Queries:")
    print("=" * 60)

    alice_idx, diana_idx = 0, 3
    is_grandparent = grandparent[alice_idx, diana_idx].item()
    print(f"\nIs {names[alice_idx]} grandparent of {names[diana_idx]}? {is_grandparent > 0.5}")

    alice_idx, eve_idx = 0, 4
    is_grandparent = grandparent[alice_idx, eve_idx].item()
    print(f"Is {names[alice_idx]} grandparent of {names[eve_idx]}? {is_grandparent > 0.5}")

    bob_idx, eve_idx = 1, 4
    is_grandparent = grandparent[bob_idx, eve_idx].item()
    print(f"Is {names[bob_idx]} grandparent of {names[eve_idx]}? {is_grandparent > 0.5}")

    print("\n" + "=" * 60)
    print("Symbolic reasoning complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
