# -*- coding: utf-8 -*-
"""
# Comp1804 Lab 1: Intro to Python

Make sure to read the accompanying slides before or alongside this tutorial.

The slides can be found on Moodle with the name "Comp1804_Lab1_python_basics.pdf"

This notebook will guide you through a basic (and practice-focused) introduction to Python.

[Adapted by  from last year's tutorial, which was adapted from [here](https://github.com/Pierian-Data/Complete-Python-3-Bootcamp/tree/master/00-Python%20Object%20and%20Data%20Structure%20Basics).]

# Import modules

One of Python's strengths is the existence of many added functionalities provided by external modules. A module is a Python script that contains various functions and classes (a class is a blueprint to create an object) that we can use. A collection of modules is called a package.

Since a module exists outside of this script it needs to be "imported", that is it needs to be made known and availaible to this script. Like this:
"""

# random is a package that lets us generate lots of different types of random numbers
import random

# To call functions defined within a package we use the "dot notation"
#
# The output of the next line should be a
print(random.random())

"""# Variables

To keep track of what's happening in the script, we need variable.
Variables are objects that contain specific information, like a number, a list of numbers or a piece of text.

They are created via assignment. That is we can use the equal sign (=) to assign whatever it's on its right hand side to the variable on its left hand side. A variable is identified by its name. To summarize, an assignment follows the structure:

`name_of_variable = object`

The equal sign is called assignment operator.

For example:
"""

# Here we create a variable called "variable_num" and assign the number 0 to it
variable_num= 0

# Here we create a variable called "variable_test" and assign the text "this is a text" to it
variable_text= "this is a text"

print(variable_num)
print(variable_text)

"""Now we can use these variables again and again and they will still have the same value unless we change it. Let's try again:"""

print(variable_num)
print(variable_text)

"""Remember to assign useful names to your variables! Try to avoid using generic names like "a" and "b". The name of a variable should make it easy to remember what is inside that variable.

Also, variable names should follow a few rules:

    1. Names should not start with a number
    2. Do not use spaces in the name, use _ instead
    3. Do not use any of these symbols :'",<>/?|\()!@#$%^&*~-+
    4. It's considered best practice (PEP8) that names are lowercase or CamelCase
    5. Do not use words that have special meaning in Python like "list" and "str"

For example:
"""

day_of_the_week = 'Friday' # good practice
a= 'Friday' # less good because it's less self-explanatory

# Notice the difference in readibility:
print('Today is', day_of_the_week)
print('Today is', a)

"""After I assign a variable, can I change it?"""

# Reassignment
variable_num = 20

# Check
print(variable_num)

"""We can change the value of a variable by performing a new assignment.
We can also use a variable to assign another variable. We can even overwrite a variable with a modified version of itself. For example:
"""

# assign a variable with a modified version of itself:
variable_num = variable_num + variable_num

# assign a variable with a variable
second_variable_num = variable_num

# Check
print(variable_num)
print(second_variable_num)

"""# Main Python objects

There are many different types of objects that can be assigned to a variable. Python has a few important data types that you will be working with all the time. Common data types include:
* **int** (for integer numbers)
* **float** (for floating-point numbers, that is numbers with decimal points)
* **str** (for string)
* **list** (a collection of objects)
* **tuple** (an immutable collection of objects)
* **dict** (for dictionary)
* **set** (a collection of unique objects)
* **bool** (for Boolean True/False)

You can check what type of object is assigned to a variable using Python's built-in `type()` function.

For example:
"""

variable_int= 1
variable_float= 1.1 #note that 1 is an integer, but 1. is a float!

print(type(variable_int))
print(type(variable_float))
print(type(variable_text)) #remember the one from before?

"""Let's dig a bit deeper into each variable type.

## Strings

Strings are used in Python to record text information. Strings in Python are actually a *sequence* of characters in a specific order. This means that the word "car" is interpreted by Python as the letter "c" in the first position, the letter "a" in the second position and the letter "r" in the third position.

### Creating strings

To create a string in Python you need to use either single quotes or double quotes. For example:
"""

# Single word
var_string='word'
print(var_string)

# Entire phrase
var_string='This is a string with multiple words'

# Note that Python doesn't really know what a word is. It simply stores a sequence
# of characters that includes white spaces
print(var_string)

# We can also use double quote
var_string="This is still a string"
print(var_string)

# Be careful with quotes within quotes!
# If we were to run the following exactly as it is it would give an error:

#' I'm using single quotes, but this will create an error'
# You can try uncommenting the line above to see the error!

# The reason for the error above is because Python interprets the single quote in <code>I'm</code> as the end of the string.
# So it doesn't know how to interpret what follows it! You can use combinations of double and single quotes to get the complete statement.
# Instead we need to write the text within double quotes, so that the apostrophe is interpreted correctly:
" I'm using single quotes, but this will create an error "

var_string="Now I'm ready to use the single quotes inside a string!"
print(var_string)

"""We can use a function called len() to check the length of a string! This function in built-in into Python and counts all characters in a string, including spaces and punctuation."""

len('Hello World')

"""We can retrieve portions (that is, individual or subsets of characters) of a strings by indexing. In Python indexing is performed by having square brackets after a variable name, where the square brackets contain the indices we are interested in.

Important note: Python's indexing starts at 0. This means that the first character has the index 0, the second has the index 1, and so on...

For example:
"""

var_string= "012345 This is a string"

# Show first element (in this case the number 0)
var_string[0]

# second character
var_string[1]

"""How about the last character? We have two ways:
1. use negative numbers, that is count the position going backwards
2. remember that the last character is in a position given by the length of the string itself **minus** 1 (because indexing starts at 0)

"""

# how about the last one?
print(var_string[-1])
print(var_string[len(var_string)-1])

"""We can use a <code>:</code> to perform *slicing* which grabs indices based on a specific pattern. The pattern is:
`start:stop:step`

It grabs everything from index `start` (included) to index `stop` (**not** included) in increments of `step`.

That is:
"""

#Check the string
print(var_string)

#This will print character 1,3,5 (not 5 because 7 is not included).
# You can double check from the string itself!
print(var_string[1:7:2])

# This will print from beginning to end in increments of 2
print(var_string[::2])

"""Note that is `start` is omitted, Python starts at the beginning. If `end` is omitted, Python will keep going until the last index. If `step` is omitted each successive index is incremented by one.
For example:
"""

# Take everything from the character at the start until the 5th one
# (note the 5th character has the index 4, while index 5, the 6th character, is excluded)
print(var_string[:5])

# Finally, we can use this to print a string backwards
var_string[::-1]

"""### String Properties
It's important to note that strings have an important property known as *immutability*. This means that once a string is created, the elements within it can not be changed or replaced. For example:
"""

var_string='try'

# If we tried to change the first letter to 'x' like in the next line it would give us an error. 
# var_string[0] = 'x'

# You can try un-commenting the line above to see the error.
# You would notice how the error tells us directly what we can't do, change the item assignment!

# Something we *can* do is concatenate strings!


var_string

# Concatenate strings!
var_string + ' concatenate me!'

# We can reassign s completely though!
var_string = var_string + ' concatenate me!'

print(var_string)

"""We can use the multiplication symbol to create repetition!"""

letter = 'z '
10*letter

"""### Basic Built-in String methods

Objects in Python usually have built-in methods. These methods are functions inside the object (we will learn about these in much more depth later) that can perform actions or commands on the object itself.

We call methods with the "dot notation", that is a period followed by the method name. Methods are in the form:

object.method(parameters)

Where parameters are extra arguments we can pass into the method. Don't worry if the details don't make 100% sense right now. Later on we will be creating our own objects and functions!

Here are some examples of built-in methods in strings:
"""

var_string

# Upper Case a string
var_string.upper()

# Lower case
var_string.lower()

# Split a string by blank space (this is the default)
var_string.split()

# Split by a specific element (doesn't include the element that was split on)
var_string.split('string')

"""Wait, what variable type is that? Well, it's a list, which we'll see shortly.

### Print Formatting

We can use the .format() method to add formatted objects to printed string statements.

The easiest way to show this is through an example:
"""

'Insert another {} string with curly brackets: {}'.format('The inserted string',"the 2nd str")

"""We will see more of this in future tutorials.

## Lists

Lists are sequences of objects, that can be of many different types. We can create lists by including the objects within square brackets. Like:
"""

var_list = [1, 2, 3, 4, 5, 6, 7, 8]
# lists have the attribute len too! They're sequences after all...
print(len(var_list))

"""Since lists are sequences, we can index them like we did for strings:"""



var_list[:3]

var_list[3:]

var_list[-2:]

"""We can add new objects at the end (and only at the end) of a list. We can do this in different ways:
1. with the method `extend`
2. with the method `append`
3. with the addition operator
"""

var_list.extend([7,8])
var_list

var_list.append(9)
var_list

var_list += [10]
var_list

"""Lists are pretty versatile and can contain any type of object. Even different types in the same list. Even a list itself..."""

new_var_list = [20, 21, 22]
list_of_lists = [var_list, new_var_list]
list_of_lists

"""We can index lists of lists by having one indexing follow the other. That is, we need to use two sets of square brackets, like this:"""

# take the index 0 from the list with index 1 (that is, the second list)
list_of_lists[1][0]

#Finally, an example of list with mixed object types
mixed_list = [0, '0', [1,'1']]
print(mixed_list)

"""## Tuples"""

#Tuples are just immutable lists. Use () instead of []
var_tuple = (1, 2, 3)
len(var_tuple)

new_var_tuple = (4, 5, 6)
new_var_tuple[2]

list_of_tuples = [var_tuple, new_var_tuple]
list_of_tuples

"""## Dictionaries

We've looked at *sequences* in Python but now we're going to switch gears and learn about *mappings* in Python. It is basically similar to hash tables or maps in other languages.

So what are mappings? Mappings are a collection of objects that are stored by a *key*, unlike a sequence that stored objects by their relative position. This is an important distinction, since mappings won't retain order since they have objects defined by a key.

A Python dictionary consists of a key and an associated value. That value can be almost any Python object.

### Constructing a Dictionary
Let's see how we can construct dictionaries to get a better understanding of how they work!
"""

# Make a dictionary with {} and : to signify a key and a value
my_dict = {'key1':'value1','key2':'value2'}

# Call values by their key
my_dict['key1']

"""Its important to note that dictionaries are very flexible in the data types they can hold. For example:"""

my_dict = {'key1':123,'key2':[12,23,33],'key3':['item0','item1','item2']}

# Let's call items from the dictionary
my_dict['key3']

# Can call an index on that value
my_dict['key2']

# Can then even call methods on that value
my_dict['key3'][0].upper()

"""We can affect the values of a key as well. For instance:"""

my_dict['key1']

# Subtract 123 from the value
my_dict['key1'] = my_dict['key1'] + 103

#Check
my_dict['key1']

"""A quick note, Python has a built-in method of doing a self subtraction or addition (or multiplication or division). We could have also used += or -= for the above statement. For example:"""

# Set the object equal to itself minus 123
my_dict['key1'] -= 123

#check
my_dict['key1']

"""### A few Dictionary Methods

There are a few methods we can call on a dictionary. Let's get a quick introduction to a few of them:
"""

# Create a typical dictionary
d = {'key1':1,'key2':2,'key3':3}

# Method to return all the keys
d.keys()

# Method to grab all values
d.values()

# Method to return tuples of all items
d.items()

"""### Another way for creating dictionaries

We can also create keys by assignment. For instance if we started off with an empty dictionary, we could continually add to it:
"""

# Like a map or hash table in other languages
captains = {}
captains["Enterprise"] = "Kirk"
captains["Enterprise D"] = "Picard"
captains["Deep Space Nine"] = "Sisko"
captains["Voyager"] = "Janeway"

print(captains["Voyager"])

print(captains.get("Enterprise"))

print(captains.get("NX-01"))

for ship in captains:
    print(ship + ": " + captains[ship])

"""# Functions

Functions are pieces of code that define a specific behaviour. They take some parameters as input and return a specific output. Functions are defined by the keyword `def` followed by the name of the function, the parameters within brackets and a colon ":". That is:


"""

# here the function is DEFINED
def SquareIt(x):
    #x is the input parameter
    # this function returns the square of the input x:
    return x ** 2

# here the function is CALLED
# a function CALL is performed using its name followed by the specific parameter
# we want to apply the function too
print(SquareIt(2))

# we can also apply functions to variables
var_num= 3
print(SquareIt(3))

"""There is lots of flexibility with functions, but we'll only mention two main features for now.

1. We can pass multiple arguments and get multiple outputs:
2. Some of the parameters can have a default value, so that even if we don't pass those parameters when we call the function, Python knows what the default value should be. However, we can still give those parameters a value different from the default.

Like this:
"""

#You can pass functions around as parameters
def MultipleOperations(x, y =2):
    # takes two parameters, returns two outputs
    return x+y, x*y

# Note that for readability we also use the name of the parameters to the function
# This way we don't have to remember the order of the parameters in the definition
print(MultipleOperations(x=3)) #should return 5 and 6
print(MultipleOperations(x=3,y=10)) #should return 13 and 30

"""# Boolean Expressions

Boolean expression are used to evaluated conditions. That is, whether something is True or False. Some common boolean operators are:
1. A==B: check if A is equal to B
2. A>B: check if A is greater than B (variants are: >=, <, <=)

We can also combine boolean expression by using operators like `and` and `or`.
"""

print(1 == 3)

print(True or False)

"""We can use Boolean operator with `if` statements, which allows us to execute different code blocks based on a given condition. For example, we may want to select the highest amongst two variables and assign that one to a new variable. Like this:"""

first_var=1
second_var=3

if first_var == second_var:
    final_var= first_var
elif first_var > second_var:
    final_var= first_var
else:
    final_var= second_var

print(final_var)

"""NOTE the colon (:) a the end of each if statement and the indented blocks! White spaces at the beginning of a line really do matter in Python. If the code was like in the comment below it would give an error:"""

"""
if first_var == second_var:
final_var= first_var
elif first_var > second_var:
 final_var= first_var
else:
ffinal_var= second_var
"""
first_var=1
second_var=3

if first_var == second_var:
    final_var= first_var
elif first_var > second_var:
    final_var= first_var
else:
    final_var= second_var

print(final_var)

"""# Loops

Finally, we can iterate over a collection of objects (like a list!) using the `for` operator. The syntax has the keyword `for`, followed by a variable that will store the value of the loop index, followed by all the values we want to iterate over, followed by the colon (:). [And don't forget the indentation block!]

"""

# iterate over all numbers from 0 to 10
for x in [0,1,2,3,4,5,6,7,8,9]:
    print(x)

"""we can replace the list [0,1,2,3,4,5,6,7,8,9] with the output of the function range(10). Specifically, the function range(x) will output all the numbers from 0 up to and **excluding** x. So:"""

for x in range(10):
    print(x)

"""Finally, we can:
1. Have `if` conditions within `for` loops
2. Control `for` loops with two keywords:
    * `continue`: skip the rest of the block of code that comes after this keyword and move to the next index in the `for` loop
    * `break`: skip the rest of the block of code that comes after this keyword and stop the whole `for` loop .

For example:
"""

# iterate from 0 to 9, ignore the number 1, print the other numbers and stops when 5 has been printed
for x in range(10):
    if x == 1:
        continue
    if x > 5:
        break
    print(x)

"""For more information about conditionals and loops, have a look [at this link](https://www.tutorialspoint.com/python/python_decision_making.htm) and [this other one](https://www.tutorialspoint.com/python/python_loops.htm).

#Activity

Write some code that creates a list of integers (any integer), loops through each element of the list, and only prints out each number multiplied by 2
"""



"""# Some notes about Python arithmetic

In Python you can do all the basic operations:
+,-,*

*   addition with +
*   subtraction with -
*   multiplication with *
*   division with /
*   Exponentiation with ** (with the square root being **0.5, try it!)
*   List item



There are also two special types of divisions:

*   Floor division: here only the integer part of the results of the division is returned
*   Modulo division: returns the remainder after the division

That is:

"""

# Floor Division
print(7//4)
#7 divided by 4 is 1.75, so its integer part is 1

# Modulo division
print(7%4)

"""Note that if you're updating a variable by performing an arithmetic operation on itself, you can use some shortcuts. Python lets you add, subtract, multiply and divide numbers with reassignment using `+=`, `-=`, `*=`, `/=` and `**=`. For example:"""

var_num= 5
var_num += 4 #this adds 10 to var_num and assigns the result to var_num
print(var_num)

var_num **= .5 #this takes the square root of var_num and assigns the result to var_num
print(var_num)