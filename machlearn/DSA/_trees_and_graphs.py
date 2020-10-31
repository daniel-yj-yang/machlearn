# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

INT_MIN = float("-infinity")
INT_MAX = float("infinity")

# A Binary tree node

class BST_Node:
    def __init__(self, value):
        self.value = value
        self.left_node = None
        self.right_node = None


def construct_Binary_Search_Tree(range_min, range_max):

    global value_index
    global preorder_traversal
    global preorder_traversal_size

    # Base Case
    if value_index >= preorder_traversal_size: return None

    root_node = None

    value = preorder_traversal[value_index]

    # If current value of pre[] is in range, then it is part of current subtree
    print(f'testing whether {range_min} < {value} <= {range_max}')

    if(range_min < value and value <= range_max):

        print(f'creating new node, with value = {value}')
        root_node = BST_Node(value)

        value_index += 1

        if(value_index < preorder_traversal_size):

            # Construct the subtree under root

            # All nodes which are in range {min.. key} will go in left subtree, and first such node will be root of left subtree
            root_node.left_node = construct_Binary_Search_Tree(range_min, value)

        if(value_index < preorder_traversal_size):

            # All nodes which are in range{key..max} will go to right subtree, and first such node will be root of right subtree
            root_node.right_node = construct_Binary_Search_Tree(value, range_max)

    return root_node



# A utility function to print inorder traversal of Binary Tree
def print_order(node, type = "Inorder"):
    if node:
        if type == "Inorder":
            print_order(node.left_node, type = type)
            print(node.value),
            print_order(node.right_node, type = type)
        elif type == "Preorder":
            print(node.value),
            print_order(node.left_node, type = type)
            print_order(node.right_node, type = type)
        elif type == "Postorder":
            print_order(node.left_node, type = type)
            print_order(node.right_node, type = type)
            print(node.value),


# Driver code
preorder_traversal = [10, 5, 1, 7, 40, 50]

# Function call
value_index = 0
preorder_traversal_size = len(preorder_traversal)
root_node = construct_Binary_Search_Tree(INT_MIN, INT_MAX)

print("Inorder traversal of the constructed tree: ")
print_order(root_node, "Inorder")

print_order(root_node, "Preorder")

print_order(root_node, "Postorder")
