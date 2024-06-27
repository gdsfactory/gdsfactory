# Report for Assignment 1

## Project chosen

Name: gdsfactory
URL: https://github.com/gdsfactory/gdsfactory
Number of lines of code and the tool used to count it:  45356
Programming language: Python

## Coverage measurement
### Coverage.py

We used coverage.py to measure coverage after installing it and its requirements.
We then cloned the original repository.
git clone https://github.com/gdsfactory/gdsfactory.git
Then we changed the directory to /gdsfactory.
cd gdsfactory
Then we installed all the required dependencies and modules.
pip install .
pip install pytest_regressions
pip install jsondiff
pip install jsonschema
Finally, we ran coverage.py.
coverage run -m pytest -s
coverage report
coverage html

### Your own coverage tool
#### Levente Zsiga - lzs201
manhattan_direction()

snap_angle()


#### Keyla Domingos Lopes - kdo208
def _parse_coordinate

Using this example input, we get the following result:

def get_min_sbend_size

Using this example we get, the following results:

#### Mike Obiekwe - mob202

polygon_grow()

remove_flat_angles()


## Coverage improvement
### Individual tests

#### Levente Zsiga - lzs201

test_manhattan_direction.py

No test covers manhattan_direction initially, so I created a new file called test_manhatten_direction.py which tests all 6 conditional branches in the function
Coverage report after adding test for manhattan_direction

The coverage improvement is 100%.

Test_snap_angle.py

Tests doesn't fully cover snap_angle, initial coverage is 67%, with current tests, only the else branch is covered so I create a new test that covers all conditional branches. New test makes sure all possible branches are covered with various inputs:
With the new test coverage is 100%, so the improvement is 33%.



#### Keyla Domingos - kdo208

test_parse_coordinate.py

The tests do not directly cover the def _parse_coordinate, which is why we created an entirely new test that exclusively covers the function and its branches.


The original branch coverage of the function was as follows:

The new branch coverage of the function after the new test is as follows:


The coverage is improved because before the introduction of this new test only the body of the elif branch was covered by tests and the other branches were not. This new test allows for the bodies of all the branches to be covered as it tests every possible case.


test_bend_s.py
The tests do not directly cover the def get_min_sbend_size, which is why we created an entirely new test that exclusively covers the function and its branches.





The original branch coverage of the function was as follows:

The new branch coverage of the function after the new test is as follows:

The new test cases cover all the branches and conditions to make sure that each path, exception handling and different input scenarios, is executed at least once to improve the overall branch coverage. The for loop, however, is not covered by these test cases.



#### Mike Obiekwe - mob202
test_polygon_grow.py
There was no test for polygon_grow, so I created one to cover to get at least 80% coverage:

The coverage obtained is 88% as it shows, so the coverage improvement is 88%


Test_remove_flat_angles.py
Similarly, there were no tests covering remove_flat_angles, so created a new one:

This time managed to get 100% coverage by creating test cases for all branches.



### Overall

<Provide a screenshot of the old coverage results by running an existing tool (the same as you already showed above)>

<Provide a screenshot of the new coverage results by running the existing tool using all test modifications made by the group>

## Statement of individual contributions
