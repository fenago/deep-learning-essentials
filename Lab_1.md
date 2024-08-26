
Lab 1: Introduction to Machine Learning with TensorFlow
========================================================



**The TensorFlow Library in Python**

TensorFlow can be used in Python by importing certain libraries. You can
import libraries in Python using the `import` statement:


```
import tensorflow as tf
```


In the preceding command, you have imported the TensorFlow library and
used the shorthand `tf`.

In the next exercise, you will learn how to import the TensorFlow
library and check its version so that you can utilize the classes and
functions supplied by the library, which is an important and necessary
first step when utilizing the library.

Exercise 1.01: Verifying Your Version of TensorFlow
---------------------------------------------------

In this exercise, you will load TensorFlow and check which version is
installed on your system.

Perform the following steps:

1.  Open a Jupyter notebook to implement this exercise by typing
    `jupyter notebook` in the terminal.

2.  Import the TensorFlow library by entering the following code in the
    Jupyter cell:
    
    ```
    import tensorflow as tf
    ```


3.  Verify the version of TensorFlow using the following command:

    
    ```
    tf.__version__
    ```


    This will result in the following output:

    
    ```
    '2.6.0'
    ```


    As you can see from the preceding output, the version of TensorFlow
    is `2.6.0`.

    Note

    The version may vary on your system if you have not set up the
    environment using the steps provided in *Preface*.

In this exercise, you successfully imported TensorFlow. You have also
checked which version of TensorFlow is installed on your system.

This task can be done for any imported library in Python and is useful
for debugging and referencing documentation.





Introduction to Tensors
=======================

In the following exercise, you will learn how to create tensors of
various ranks using TensorFlow\'s `Variable` class.

Exercise 1.02: Creating Scalars, Vectors, Matrices, and Tensors in TensorFlow
-----------------------------------------------------------------------------

The votes cast for different candidates of three different political
parties in districts A and B are as follows:

![](./images/B16341_01_03.jpg)




You are required to do the following:

-   Create a scalar to store the votes cast for `Candidate 1`
    of political party `X` in district `A`, that is,
    `4113`, and check its shape and rank.
-   Create a vector to represent the proportion of votes cast for three
    different candidates of political party `X` in district
    `A` and check its shape and rank.
-   Create a matrix to represent the votes cast for three different
    candidates of political parties `X` and `Y` and
    check its shape and rank.
-   Create a tensor to represent the votes cast for three different
    candidates in two different districts, for three political parties,
    and check its shape and rank.

Perform the following steps to complete this exercise:

1.  Import the TensorFlow library:
    
    ```
    import tensorflow as tf
    ```


2.  Create an integer variable using TensorFlow\'s `Variable`
    class and pass `4113` to represent the number of votes
    cast for a particular candidate. Also, pass `tf.int16` as
    a second argument to ensure that the input number is an integer
    datatype. Print the result:

    Note

    The datatype does not have to be explicitly defined. If one is not
    defined, the datatype will be determined by TensorFlow\'s
    `convert_to_tensor` function.

    
    ```
    int_variable = tf.Variable(4113, tf.int16)
    int_variable
    ```


    This will result in the following output:

    
    ```
    <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=4113>
    ```


    Here, you can see the attributes of the variable created, including
    the name, `Variable:0`, the shape, datatype, and the NumPy
    representation of the tensor.

3.  Use TensorFlow\'s `rank` function to print the rank of the
    variable created:

    
    ```
    tf.rank(int_variable)
    ```


    This will result in the following output:

    
    ```
    <tf.Tensor: shape=(), dtype=int32, numpy=0>
    ```


    You can see that the rank of the integer variable that was created
    is `0` from the NumPy representation of the tensor.

4.  Access the integer variable of the rank by calling the
    `numpy` attribute:

    
    ```
    tf.rank(int_variable).numpy()
    ```


    This will result in the following output:

    
    ```
    0
    ```


    The rank of the scalar is `0`.

    Note

    All attributes of the result of the `rank` function can be
    called, including the `shape` and `dtype`
    attributes.

5.  Call the `shape` attribute of the integer to find the
    shape of the tensor:

    
    ```
    int_variable.shape
    ```


    This will result in the following output:

    
    ```
    TensorShape([])
    ```


    The preceding output signifies that the shape of the tensor has no
    size, which is representative of a scalar.

6.  Print the `shape` of the scalar variable as a Python list:

    
    ```
    int_variable.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    []
    ```


7.  Create a `vector` variable using TensorFlow\'s
    `Variable` class. Pass a list for the vector to represent
    the proportion of votes cast for three different candidates, and
    pass in a second argument for the datatype as `tf.float32`
    to ensure that it is a `float` datatype. Print the result:

    
    ```
    vector_variable = tf.Variable([0.23, 0.42, 0.35], \
                                  tf.float32)
    vector_variable
    ```


    This will result in the following output:

    
    ```
    <tf.Variable 'Variable:0' shape(3,) dtype=float32, 
    numpy=array([0.23, 0.42, 0.35], dtype=float32)>
    ```


    You can see that the shape and NumPy attributes are different from
    the scalar variable created earlier. The shape is now
    `(3,)`, indicating that the tensor is one-dimensional with
    three elements along that dimension.

8.  Print the rank of the `vector` variable using
    TensorFlow\'s `rank` function as a NumPy variable:

    
    ```
    tf.rank(vector_variable).numpy()
    ```


    This will result in the following output:

    
    ```
    1
    ```


    Here, you can see that the rank of the vector variable is
    `1`, confirming that this variable is one-dimensional.

9.  Print the shape of the `vector` variable as a Python list:

    
    ```
    vector_variable.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    [3]
    ```


10. Create a matrix variable using TensorFlow\'s `Variable`
    class. Pass a list of lists of integers for the matrix to represent
    the votes cast for three different candidates in two different
    districts. This matrix will have three columns representing the
    candidates, and two rows representing the districts. Pass in a
    second argument for the datatype as `tf.int32` to ensure
    that it is an integer datatype. Print the result:

    
    ```
    matrix_variable = tf.Variable([[4113, 7511, 6259], \
                                   [3870, 6725, 6962]], \
                                  tf.int32)
    matrix_variable
    ```


    This will result in the following output:

    
    ![](./images/B16341_01_04.jpg)




11. Print the rank of the matrix variable as a NumPy variable:

    
    ```
    tf.rank(matrix_variable).numpy()
    ```


    This will result in the following output:

    
    ```
    2
    ```


    Here, you can see that the rank of the matrix variable is
    `2`, confirming that this variable is two-dimensional.

12. Print the shape of the matrix variable as a Python list:

    
    ```
    matrix_variable.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    [2, 3]
    ```


13. Create a tensor variable using TensorFlow\'s `Variable`
    class. Pass in a triple nested list of integers for the tensor to
    represent the votes cast for three different candidates in two
    different districts, for three political parties. Print the result:

    
    ```
    tensor_variable = tf.Variable([[[4113, 7511, 6259], \
                                    [3870, 6725, 6962]], \
                                   [[5102, 7038, 6591], \
                                    [3661, 5901, 6235]], \
                                   [[951, 1208, 1098], \
                                    [870, 645, 948]]])
    tensor_variable
    ```


    This will result in the following output:

    
    ![](./images/B16341_01_05.jpg)




14. Print the rank of the tensor variable as a NumPy variable:

    
    ```
    tf.rank(tensor_variable).numpy()
    ```


    This will result in the following output:

    
    ```
    3
    ```


    Here, you can see that the rank of the tensor variable is
    `3`, confirming that this variable is three-dimensional.

15. Print the shape of the tensor variable as a Python list:

    
    ```
    tensor_variable.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    [3, 2, 3]
    ```


    The result shows that the shape of the resulting tensor is a list
    object.

In this exercise, you have successfully created tensors of various ranks
from political voting data using TensorFlow\'s `Variable`
class. First, you created scalars, which are tensors that have a rank of
`0`. Next, you created vectors, which are tensors with a rank
of `1`. Matrices were then created, which are tensors of rank
`2`. Finally, tensors were created that have rank
`3` or more. You confirmed the rank of the tensors you created
by using TensorFlow\'s `rank` function and verified their
shape by calling the tensor\'s `shape` attribute.

In the next section, you will combine tensors to create new tensors
using tensor addition.





Tensor Addition
===============


Tensors can be added together to create new tensors. You will use the
example of matrices in this lab, but the concept can be extended to
tensors with any rank. Matrices may be added to scalars, vectors, and
other matrices under certain conditions in a process known as
broadcasting. Broadcasting refers to the process of array arithmetic on
tensors of different shapes.

In the following exercise, you will perform tensor addition on scalars,
vectors, and matrices in TensorFlow.

Exercise 1.03: Performing Tensor Addition in TensorFlow
-------------------------------------------------------

The votes cast for different candidates of three different political
parties in districts A and B are as follows:

![](./images/B16341_01_08.jpg)


Your requisite tasks are as follows:

-   Store the total number of votes cast for political party X in
    district A.
-   Store the total number of votes cast for each political party in
    district A.
-   Store the total number of votes cast for each political party in
    both districts.

Perform the following steps to complete the exercise:

1.  Import the TensorFlow library:
    
    ```
    import tensorflow as tf
    ```


2.  Create three scalar variables using TensorFlow\'s
    `Variable` class to represent the votes cast for three
    candidates of political party X in district A:
    
    ```
    int1 = tf.Variable(4113, tf.int32)
    int2 = tf.Variable(7511, tf.int32)
    int3 = tf.Variable(6529, tf.int32)
    ```


3.  Create a new variable to store the total number of votes cast for
    political party X in district A:
    
    ```
    int_sum = int1+int2+int3
    ```


4.  Print the result of the sum of the two variables as a NumPy
    variable:

    
    ```
    int_sum.numpy()
    ```


    This will result in the following output:

    
    ```
    18153
    ```


5.  Create three vectors to represent the number of votes cast for
    different political parties in district A, each with one row and
    three columns:
    
    ```
    vec1 = tf.Variable([4113, 3870, 5102], tf.int32)
    vec2 = tf.Variable([7511, 6725, 7038], tf.int32)
    vec3 = tf.Variable([6529, 6962, 6591], tf.int32)
    ```


6.  Create a new variable to store the total number of votes for each
    political party in district A:
    
    ```
    vec_sum = vec1 + vec2 + vec3
    ```


7.  Print the result of the sum of the two variables as a NumPy array:

    
    ```
    vec_sum.numpy()
    ```


    This will result in the following output:

    
    ```
    array([18153, 17557, 18731])
    ```


8.  Verify that the vector addition is as expected by performing the
    addition of each element of the vector:

    
    ```
    print((vec1[0] + vec2[0] + vec3[0]).numpy())
    print((vec1[1] + vec2[1] + vec3[1]).numpy())
    print((vec1[2] + vec2[2] + vec3[2]).numpy())
    ```


    This will result in the following output:

    
    ```
    18153
    17557
    18731
    ```


    You can see that the `+` operation on three vectors is
    simply element-wise addition of the vectors.

9.  Create three matrices to store the votes cast for candidates of each
    political party in each district:
    
    ```
    matrix1 = tf.Variable([[4113, 3870, 5102], \
                           [3611, 951, 870]], tf.int32)
    matrix2 = tf.Variable([[7511, 6725, 7038], \
                           [5901, 1208, 645]], tf.int32)
    matrix3 = tf.Variable([[6529, 6962, 6591], \
                           [6235, 1098, 948]], tf.int32)
    ```


10. Verify that the three tensors have the same shape:

    
    ```
    matrix1.shape == matrix2.shape == matrix3.shape
    ```


    This will result in the following output:

    
    ```
    True
    ```


11. Create a new variable to store the total number of votes cast for
    each political party in both districts:
    
    ```
    matrix_sum = matrix1 + matrix2 + matrix3
    ```


12. Print the result of the sum of the two variables as a NumPy array:

    
    ```
    matrix_sum.numpy()
    ```


    This will result in the following output representing the total
    votes for each candidate and each party across districts:

    
    ![](./images/B16341_01_09.jpg)



13. Verify that the tensor addition is as expected by performing the
    addition of each element of the vector:

    
    ```
    print((matrix1[0][0] + matrix2[0][0] + matrix3[0][0]).numpy())
    print((matrix1[0][1] + matrix2[0][1] + matrix3[0][1]).numpy())
    print((matrix1[0][2] + matrix2[0][2] + matrix3[0][2]).numpy())
    print((matrix1[1][0] + matrix2[1][0] + matrix3[1][0]).numpy())
    print((matrix1[1][1] + matrix2[1][1] + matrix3[1][1]).numpy())
    print((matrix1[1][2] + matrix2[1][2] + matrix3[1][2]).numpy())
    ```


    This will result in the following output:

    
    ```
    18153
    17557
    18731
    15747
    3257
    2463
    ```


    You can see that the `+` operation is equivalent to the
    element-wise addition of the three matrices created.

In this exercise, you successfully performed tensor addition on data
representing votes cast for political candidates. The transformation can
be applied by using the `+` operation. You also verified that
addition is performed element by element, and that one way to ensure
that the transformation is valid is for the tensors to have the same
rank and shape.

In the following activity, you will further practice tensor addition in
TensorFlow.

Activity 1.01: Performing Tensor Addition in TensorFlow
-------------------------------------------------------

You work in a company that has three locations, each with two
salespersons and each location sells three products. You are required to
sum the tensors to represent the total revenue for each product across
locations.

![](./images/B16341_01_10.jpg)



The steps you will take are as follows:

1.  Import the TensorFlow library.

2.  Create two scalars to represent the total revenue for
    `Product A` by all salespeople at `Location X`
    using TensorFlow\'s `Variable` class. The first variable
    will have a value of `2706` and the second will have a
    value of `2386`.

3.  Create a new variable as the sum of the scalars and print the
    result.

    You should get the following output:

    
    ```
    5092
    ```


4.  Create a vector with values `[2706, 2799, 5102]` and a
    scalar with the value `95` using TensorFlow\'s
    `Variable` class.

5.  Create a new variable as the sum of the scalar with the vector to
    represent the sales goal for `Salesperson 1` at
    `Location X` and print the result.

    You should get the following output:

    
    ![](./images/B16341_01_11.jpg)



6.  Create three tensors with a rank of 2 representing the revenue for
    each salesperson, product, and location using TensorFlow\'s
    `Variable` class. The first tensor will have the value
    `[[2706, 2799, 5102], [2386, 4089, 5932]]`, the second
    will have the value
    `[[5901, 1208, 645], [6235, 1098, 948]]`, and the third
    will have `[[3908, 2339, 5520], [4544, 1978, 4729]]`.

7.  Create a new variable as the sum of the matrices and print the
    result:
    
    ![](./images/B16341_01_12.jpg)



In the following section, you will learn how to change a tensor\'s shape
and rank.





Reshaping
=========

Some operations, such as addition, can only be applied to tensors if
they meet certain conditions. Reshaping is one method for modifying the
shape of tensors so that such operations can be performed. Reshaping
takes the elements of a tensor and rearranges them into a tensor of a
different size. A tensor of any size can be reshaped so long as the
number of total elements remains the same.


In the following exercise, reshaping and transposition are demonstrated
on tensors using TensorFlow.

Exercise 1.04: Performing Tensor Reshaping and Transposition in TensorFlow
--------------------------------------------------------------------------

In this exercise, you will learn how to perform tensor reshaping and
transposition using the TensorFlow library.

Perform the following steps:

1.  Import the TensorFlow library and create a matrix with two rows and
    four columns using TensorFlow\'s `Variable` class:
    
    ```
    import tensorflow as tf
    matrix1 = tf.Variable([[1,2,3,4], [5,6,7,8]])
    ```


2.  Verify the shape of the matrix by calling the `shape`
    attribute of the matrix as a Python list:

    
    ```
    matrix1.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    [2, 4]
    ```


    You see that the shape of the matrix is `[2,4]`.

3.  Use TensorFlow\'s `reshape` function to change the matrix
    to a matrix with four rows and two columns by passing in the matrix
    and the desired new shape:

    
    ```
    reshape1 = tf.reshape(matrix1, shape=[4, 2])
    reshape1
    ```


    You should get the following output:

    
    ![](./images/B16341_01_16.jpg)




4.  Verify the shape of the reshaped matrix by calling the
    `shape` attribute as a Python list:

    
    ```
    reshape1.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    [4, 2]
    ```


    Here, you can see that the shape of the matrix has changed to your
    desired shape, `[4,2]`.

5.  Use TensorFlow\'s `reshape` function to convert the matrix
    into a matrix with one row and eight columns. Pass the matrix and
    the desired new shape as parameters to the `reshape`
    function:

    
    ```
    reshape2 = tf.reshape(matrix1, shape=[1, 8])
    reshape2
    ```


    You should get the following output:

    
    ```
    <tf.Tensor: shape=(1, 8), dtype=int32, numpy=array([[1, 2, 3, 4, 5, 6, 7, 8]])>
    ```


6.  Verify the shape of the reshaped matrix by calling the
    `shape` attribute as a Python list:

    
    ```
    reshape2.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    [1, 8]
    ```


    The preceding output confirms the shape of the reshaped matrix as
    `[1, 8]`.

7.  Use TensorFlow\'s `reshape` function to convert the matrix
    into a matrix with eight rows and one column, passing the matrix and
    the desired new shape as parameters to the `reshape`
    function:

    
    ```
    reshape3 = tf.reshape(matrix1, shape=[8, 1])
    reshape3
    ```


    You should get the following output:

    
    ![](./images/B16341_01_17.jpg)




8.  Verify the shape of the reshaped matrix by calling the
    `shape` attribute as a Python list:

    
    ```
    reshape3.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    [8, 1]
    ```


    The preceding output confirms the shape of the reshaped matrix as
    `[8, 1]`.

9.  Use TensorFlow\'s `reshape` function to convert the matrix
    to a tensor of size `2x2x2`. Pass the matrix and the
    desired new shape as parameters to the reshape function:

    
    ```
    reshape4 = tf.reshape(matrix1, shape=[2, 2, 2])
    reshape4
    ```


    You should get the following output:

    
    ![](./images/B16341_01_18.jpg)




10. Verify the shape of the reshaped matrix by calling the
    `shape` attribute as a Python list:

    
    ```
    reshape4.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    [2, 2, 2]
    ```


    The preceding output confirms the shape of the reshaped matrix as
    `[2, 2, 2]`.

11. Verify that the rank has changed using TensorFlow\'s
    `rank` function and print the result as a NumPy variable:

    
    ```
    tf.rank(reshape4).numpy()
    ```


    This will result in the following output:

    
    ```
    3
    ```


12. Use TensorFlow\'s `transpose` function to convert the
    matrix of size `2X4` to a matrix of size `4x2`:

    
    ```
    transpose1 = tf.transpose(matrix1)
    transpose1
    ```


    You should get the following output:

    
    ![](./images/B16341_01_19.jpg)




13. Verify that the `reshape` function and the
    `transpose` function create different resulting matrices
    when applied to the given matrix:

    
    ```
    transpose1 == reshape1
    ```


    
    ![](./images/B16341_01_20.jpg)



14. Use TensorFlow\'s `transpose` function to transpose the
    reshaped matrix in *step 9*:

    
    ```
    transpose2 = tf.transpose(reshape4)
    transpose2
    ```


    This will result in the following output:

    
    ![](./images/B16341_01_21.jpg)



This result shows how the resulting tensor appears after reshaping and
transposing a tensor.

In this exercise, you have successfully modified the shape of a tensor
either through reshaping or transposition. You studied how the shape and
rank of the tensor changes following the reshaping and transposition
operation.

In the following activity, you will test your knowledge on how to
reshape and transpose tensors using TensorFlow.

Activity 1.02: Performing Tensor Reshaping and Transposition in TensorFlow
--------------------------------------------------------------------------

In this activity, you are required to simulate the grouping of 24 school
children for class projects. The dimensions of each resulting reshaped
or transposed tensor will represent the size of each group.

Perform the following steps:

1.  Import the TensorFlow library.

2.  Create a one-dimensional tensor with 24 monotonically increasing
    elements using the `Variable` class to represent the IDs
    of the school children. Verify the shape of the matrix.

    You should get the following output:

    
    ```
    [24]
    ```


3.  Reshape the matrix so that it has 12 rows and 2 columns using
    TensorFlow\'s `reshape` function representing 12 pairs of
    school children. Verify the shape of the new matrix.

    You should get the following output:

    
    ```
    [12, 2]
    ```


4.  Reshape the original matrix so that it has a shape of
    `3x4x2` using TensorFlow\'s `reshape` function
    representing 3 groups of 4 sets of pairs of school children. Verify
    the shape of the new tensor.

    You should get the following output:

    
    ```
    [3, 4, 2]
    ```


5.  Verify that the rank of this new tensor is `3`.

6.  Transpose the tensor created in *step 3* to represent 2 groups of 12
    students using TensorFlow\'s `transpose` function. Verify
    the shape of the new tensor.

    You should get the following output:

    
    ```
    [2, 12]
    ```


    Note

    The solution to this activity can be found via [this
    link].

In this section, you were introduced to some of the basic components of
ANNs---tensors. You also learned about some basic manipulation of
tensors, such as addition, transposition, and reshaping. You implemented
these concepts by using functions in the TensorFlow library.

In the next topic, you will extend your understanding of linear
transformations by covering another important transformation related to
ANNs---tensor multiplication.





Tensor Multiplication
=====================


Tensor multiplication is another fundamental operation that is used
frequently in the process of building and training ANNs since
information propagates through the network from the inputs to the result
via a series of additions and multiplications. While the rules for
addition are simple and intuitive, the rules for tensors are more
complex. Tensor multiplication involves more than simple element-wise
multiplication of the elements. Rather, a more complicated procedure is
implemented that involves the dot product between the entire
rows/columns of each of the tensors to calculate each element of the
resulting tensor. This section will explain how multiplication works for
two-dimensional tensors or matrices. However, tensors of higher orders
can also be multiplied.


In the following exercise, you will perform tensor multiplication using
the TensorFlow library.

Exercise 1.05: Performing Tensor Multiplication in TensorFlow
-------------------------------------------------------------

In this exercise, you will perform tensor multiplication in TensorFlow
using TensorFlow\'s `matmul` function and the `@`
operator. In this exercise, you will use the example of data from a
sandwich retailer representing the ingredients of various sandwiches and
the costs of different ingredients. You will use matrix multiplication
to determine the costs of each sandwich.

**Sandwich recipe**:

![](./images/B16341_01_27.jpg)




**Ingredient details**:

![](./images/B16341_01_28.jpg)




**Sales projections**:

![](./images/B16341_01_29.jpg)




Perform the following steps:

1.  Import the TensorFlow library:
    
    ```
    import tensorflow as tf
    ```


2.  Create a matrix representing the different sandwich recipes, with
    the rows representing the three different sandwich offerings and the
    columns representing the combination and number of the five
    different ingredients using the `Variable` class:

    
    ```
    matrix1 = tf.Variable([[1.0,0.0,3.0,1.0,2.0], \
                           [0.0,1.0,1.0,1.0,1.0], \
                           [2.0,1.0,0.0,2.0,0.0]], \
                          tf.float32)
    matrix1
    ```


    You should get the following output:

    
    ![](./images/B16341_01_30.jpg)



3.  Verify the shape of the matrix by calling the `shape`
    attribute of the matrix as a Python list:

    
    ```
    matrix1.shape.as_list()
    ```


    This will result in the following output:

    
    ```
    [3, 5]
    ```


4.  Create a second matrix representing the cost and weight of each
    individual ingredient in which the rows represent the five
    ingredients, and the columns represent the cost and weight of the
    ingredients in grams:

    
    ```
    matrix2 = tf.Variable([[0.49, 103], \
                           [0.18, 38], \
                           [0.24, 69], \
                           [1.02, 75], \
                           [0.68, 78]])
    matrix2
    ```


    You should get the following result:

    
    ![](./images/B16341_01_31.jpg)



5.  Use TensorFlow\'s `matmul` function to perform the matrix
    multiplication of `matrix1` and `matrix2`:

    
    ```
    matmul1 = tf.matmul(matrix1, matrix2)
    matmul1
    ```


    This will result in the following output:

    
    ![](./images/B16341_01_32.jpg)




6.  Create a matrix to represent the sales projections of five different
    stores for each of the three sandwiches:
    
    ```
    matrix3 = tf.Variable([[120.0, 100.0, 90.0], \
                           [30.0, 15.0, 20.0], \
                           [220.0, 240.0, 185.0], \
                           [145.0, 160.0, 155.0], \
                           [330.0, 295.0, 290.0]])
    ```


7.  Multiply `matrix3` by the result of the matrix
    multiplication of `matrix1` and `matrix2` to
    give the expected cost and weight for each of the five stores:

    
    ```
    matmul3 = matrix3 @ matmul1
    matmul3
    ```


    This will result in the following output:

    
    ![](./images/B16341_01_33.jpg)




The resulting tensor from the multiplication shows the expected cost of
sandwiches and the expected weight of the total ingredients for each of
the stores.

In this exercise, you have successfully learned how to perform matrix
multiplication in TensorFlow using several operators. You used
TensorFlow\'s `matmul` function, as well as the shorthand
`@` operator. Each will perform the multiplication; however,
the `matmul` function has several different arguments that can
be passed into the function that make it more flexible.

Note

You can read more about the `matmul` function here:
[https://www.tensorflow.org/api\_docs/python/tf/linalg/matmul].

In the next section, you will explore some other mathematical concepts
that are related to ANNs. You will explore forward and backpropagation,
as well as activation functions.






Activity 1.03: Applying Activation Functions
--------------------------------------------

In this activity, you will recall many of the concepts used throughout
the lab as well as apply activation functions to tensors. You will
use example data of car dealership sales, apply these concepts, show the
sales records of various salespeople, and highlight those with net
positive sales.

**Sales records**:

![](./images/B16341_01_37.jpg)




**Vehicle MSRPs**:

![](./images/B16341_01_38.jpg)




**Fixed costs**:

![](./images/B16341_01_39.jpg)




Perform the following steps:

1.  Import the TensorFlow library.

2.  Create a `3x4` tensor as an input with the values
    `[[-0.013, 0.024, 0.06, 0.022], [0.001, -0.047, 0.039, 0.016], [0.018, 0.030, -0.021, -0.028]]`.
    The rows in this tensor represent the sales of various sales
    representatives, the columns represent various vehicles available at
    the dealership, and values represent the average percentage
    difference from MSRP. The values are positive or negative depending
    on whether the salesperson was able to sell for more or less than
    the MSRP.

3.  Create a `4x1` weights tensor with the shape
    `4x1` with the values
    `[[19995.95], [24995.50], [36745.50], [29995.95]]`
    representing the MSRP of the cars.

4.  Create a bias tensor of size `3x1` with the values
    `[[-2500.0], [-2500.0], [-2500.0]]` representing the fixed
    costs associated with each salesperson.

5.  Matrix multiply the input by the weight to show the average
    deviation from the MSRP on all cars and add the bias to subtract the
    fixed costs of the salesperson. Print the result.

    You should get the following result:

    
    ![](./images/B16341_01_40.jpg)




6.  Apply a ReLU activation function to highlight the net-positive
    salespeople and print the result.

    You should get the following result:

    
    ![](./images/B16341_01_41.jpg)




In subsequent chapters, you will see how to add activation functions to
your ANNs, either between layers or applied directly after a layer when
layers are defined. You will learn how to choose which activation
functions are most appropriate, which is often by hyperparameter
optimization techniques. The activation function is one example of a
hyperparameter, a parameter set before the learning process begins, that
can be tuned to find the optimal values for model performance.





Summary
=======


In this lab, you were introduced to the TensorFlow library. You
learned how to use it in the Python programming language. You created
the building blocks of ANNs (tensors) with various ranks and shapes,
performed linear transformations on tensors using TensorFlow, and
implemented addition, reshaping, transposition, and multiplication on
tensors---all of which are fundamental for understanding the underlying
mathematics of ANNs.
