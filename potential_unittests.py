# re: "least_edge"...
import numpy as np
import unittest

class TestLeastEdge(unittest.TestCase):
    def setUp(self):
        self.E = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_least_edge(self):
        least_E, dirs = least_edge(self.E)
        # Check the shape of the output arrays
        self.assertEqual(least_E.shape, self.E.shape)
        self.assertEqual(dirs.shape, self.E.shape)
        # Check the values in the last row of least_E
        self.assertTrue(np.array_equal(least_E[-1, :], self.E[-1, :]))
        # Add more assertions here to check the values in least_E and dirs

if __name__ == '__main__':
    unittest.main()

#Explanations:
    # setUp method is used to set up the test environment It’s run before each test method, and it initializes the input array E
    # The test_least_edge method is a test case for the least_edge function. It calls least_edge with E as the argument and checks the shape and values of the output arrays
        # can add more assertions in test_least_edge to check the values in least_E and dirs
    
