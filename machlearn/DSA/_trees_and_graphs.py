# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

class BST_node():
    # BST data structure
    def __init__(self, value):
        self.value = value
        self.left_node = None
        self.right_node = None

    def insert(self, value):
        if self.value:
            if value < self.value:
                if self.left_node is None:
                    self.left_node = BST_node(value)
                else:
                    self.left_node.insert(value)
            else:
                if self.right_node is None:
                    self.right_node = BST_node(value)
                else:
                    self.right_node.insert(value)
        else:
            self.value = value


class binary_search_tree:

    def __init__(self):
        pass

    def construct_BST(self, array):
        root_node = BST_node(array[0])
        for i in range(1, len(array)):
            root_node.insert(array[i])
        return root_node

    def tree_sort(self, array):
        if len(array) == 0:
            return array
        root_node = self.construct_BST(array)
        return_array = []
        self.order(root_node, return_array, type="Inorder")
        return return_array

    def construct_BST_from_preorder_traversal(self, preorder_traversal):
        """
        Time complexity: O(n)

        Reference: https://www.geeksforgeeks.org/construct-bst-from-given-preorder-traversal-set-2/
        """

        # The first element of preorder_traversal[] is the root node
        root_node = BST_node(preorder_traversal[0])

        stack = []
        stack.append(root_node)

        # Iterate through rest of the size-1 items of given preorder array
        for i in range(1, len(preorder_traversal)):
            temp_node = None

            # Keep on popping while the next value is greater than stack's top value.
            while len(stack) > 0 and preorder_traversal[i] > stack[-1].value:
                temp_node = stack.pop()

            # Make this greater value as the right child and append it to the stack
            if temp_node != None:
                temp_node.right_node = BST_node(preorder_traversal[i])
                stack.append(temp_node.right_node)

            # If the next value is less than the stack's top value, make this value as the left child of the stack's top node. append the new node to stack
            else:
                temp_node = stack[-1]
                temp_node.left_node = BST_node(preorder_traversal[i])
                stack.append(temp_node.left_node)

        return root_node

    def construct_BST_from_preorder_traversal_v2(self, range_min=float("-infinity"), range_max=float("infinity")):
        """
        Time complexity: O(n)

        Reference: https://www.geeksforgeeks.org/construct-bst-from-given-preorder-traversa/
        """

        global BST_value_index
        global preorder_traversal
        global preorder_traversal_size

        # Base Case
        if BST_value_index >= preorder_traversal_size:
            return None

        root_node = None

        value = preorder_traversal[BST_value_index]

        # If current value of pre[] is in range, then it is part of current subtree
        print(f'testing whether {range_min} < {value} <= {range_max}')

        if range_min < value and value <= range_max:

            print(f'creating new node, with value = {value}')
            root_node = BST_node(value)

            BST_value_index += 1

            # Construct the subtree under root
            if BST_value_index < preorder_traversal_size:

                # All nodes which are in range {min.. key} will go in left subtree, and first such node will be root of left subtree
                root_node.left_node = self.construct_BST_from_preorder_traversal_v2(
                    range_min, value)

            if BST_value_index < preorder_traversal_size:

                # All nodes which are in range{key..max} will go to right subtree, and first such node will be root of right subtree
                root_node.right_node = self.construct_BST_from_preorder_traversal_v2(
                    value, range_max)

        return root_node

    def order(self, root_node, return_array, type="Inorder"):
        # Recursive travesal
        if root_node:
            if type == "Inorder":
                self.order(root_node.left_node, return_array, type=type)
                return_array.append(root_node.value)
                self.order(root_node.right_node, return_array, type=type)

            elif type == "Preorder":
                return_array.append(root_node.value)
                self.order(root_node.left_node, return_array, type=type)
                self.order(root_node.right_node, return_array, type=type)

            elif type == "Postorder":
                self.order(root_node.left_node, return_array, type=type)
                self.order(root_node.right_node, return_array, type=type)
                return_array.append(root_node.value)


def tree_demo():
    BST = binary_search_tree()

    # Construct BST

    #global BST_value_index
    #global preorder_traversal
    #global preorder_traversal_size
    #BST_value_index = 0
    #preorder_traversal = [10, 5, 1, 7, 40, 50]
    #preorder_traversal_size = len(preorder_traversal)
    #root_node = BST.construct_BST_from_preorder_traversal_v2()

    #preorder_traversal = [10, 5, 1, 7, 40, 50]
    #root_node = BST.construct_BST_from_preorder_traversal(preorder_traversal)

    preorder_traversal = [10, 5, 1, 7, 40, 50]
    root_node = BST.construct_BST(preorder_traversal)

    # Using the BST

    print("\nTraversal of the constructed BST: ")

    array = []
    BST.order(root_node, array, "Inorder")
    print(f"\nInorder: {array}")

    array = []
    BST.order(root_node, array, "Preorder")
    print(f"\nPreorder: {array}")

    array = []
    BST.order(root_node, array, "Postorder")
    print(f"\nPostorder: {array}")
