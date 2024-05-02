# mm-wave Lab Physics Models

This is our repository of shared code for calculations of 
interest to mm-wave Lab. 

For a minimal working example of the format and function we are going for, 
please see the "/models/gatefidelity.py", "/plotters/plot_gatefidelity.py", 
and "/notebooks/GateErrors.ipynb" collection.

## Introduction
The idea is that all calculation-heavy code will be placed in the /models 
directory. These are calculations that everyone will use; therefore, the 
source code should be re-used and scrutinized often to check for factors of 
$2\pi$, sign errors, etc. Use these models for larger calculations and plots,
and everyone will understand where the underlying code came from, there will 
be fewer silly mistakes, and hence shorter discussions and meetings!

If you make a very complex plot, or multiple plots for the same model, you 
should write a Python file that contains your plotting functions and 
avoid cluttered Jupyter notebooks. These files should be named with the 
prefix "plot_X.py" and placed in the /plotters directory. 

Finally, your stunning Jupyter notebook can go in the /notebooks 
directory. The notebook should call functions and classes that are 
implemented either in /plotters or in /models to benefit from the shared 
(error-checked) code base. These notebooks can be individualized to 
your specific use case, but they should still be synced to the Github repo. 

You can edit the Python code in the repo and your Jupyter notebook 
simultaneously. Restart your Jupyter kernel to reload any Python code you 
edit in your Python IDE.

## Contributing
Whenever you want to edit or add to the repo, checkout a new Git branch.

Make your edits, making commits and pushing code to your branch as 
you go.

Create a pull request for your branch. One colleague needs to review and 
approve your code before you can merge your branch. Consider using the 
automated PyLinter to edit your code to match the PEP8 style guide. 

Once your code is approved, merge commit the branch and then delete it. 

Repeat as necessary.
