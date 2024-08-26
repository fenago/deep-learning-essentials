
Lab 2: Loading and Processing Data
==================================

In this lab, you will learn how to load and process a variety of
data types for modeling in TensorFlow. You will implement methods to
input data into TensorFlow models so that model training can be
optimized.

By the end of this lab, you will know how to input tabular data,
images, text, and audio data and preprocess them so that they are
suitable for training TensorFlow models.



In the first exercise, you will learn how to use pandas and scikit-learn
to load a dataset and preprocess it so that it is suitable for modeling.

Exercise 2.01: Loading Tabular Data and Rescaling Numerical Fields
------------------------------------------------------------------

The dataset, `Bias_correction_ucl.csv`, contains information
for bias correction of the next-day maximum and minimum air temperature
forecast for Seoul, South Korea. The fields represent temperature
measurements of the given date, the weather station at which the metrics
were measured, model forecasts of weather-related metrics such as
humidity, and projections for the temperature of the following day. You
are required to preprocess the data to make all the columns normally
distributed with a mean of `0` and a standard deviation of
`1`. You will demonstrate the effects with the
`Present_Tmax` column, which represents the maximum
temperature on the given date at a given weather station.

Note

The dataset can be found here: [https://github.com/fenago/deep-learning-essentials/blob/main/Lab02/Datasets/Bias_correction_ucl.csv].

Perform the following steps to complete this exercise:

1.  Open a new Jupyter notebook to implement this exercise. Save the
    file as `Exercise2-01.ipnyb`.

2.  In a new Jupyter Notebook cell, import the pandas library, as
    follows:

    
    ```
    import pandas as pd
    ```



3.  Create a new pandas DataFrame named `df` and read the
    `Bias_correction_ucl.csv` file into it. Examine whether
    your data is properly loaded by printing the resultant DataFrame:

    
    ```
    df = pd.read_csv('Bias_correction_ucl.csv')
    df
    ```


    Note

    Make sure you change the path (highlighted) to the CSV file based on
    its location on your system. If you\'re running the Jupyter notebook
    from the same directory where the CSV file is stored, you can run
    the preceding code without any modification.

    The output will be as follows:

    
![](./images/B16341_02_05.jpg)




4.  Drop the `date` column using the `drop` method
    of the DataFrame and pass in the name of the column. The
    `date` column will be dropped as it is a non-numerical
    field and rescaling will not be possible when non-numerical fields
    exist. Since you are dropping a column, both the `axis=1`
    argument and the `inplace=True` argument should be passed:
    
    ```
    df.drop('Date', inplace=True, axis=1)
    ```


5.  Plot a histogram of the `Present_Tmax` column that
    represents the maximum temperature across dates and weather stations
    within the dataset:

    
    ```
    ax = df['Present_Tmax'].hist(color='gray')
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Frequency")
    ```


    The output will be as follows:

    
![](./images/B16341_02_06.jpg)



    The resultant histogram shows the distribution of values for the
    `Present_Tmax` column. You can see that the temperature
    values vary from 20 to 38 degrees Celsius. Plotting a histogram of
    the feature values is a good way to view the distribution of values
    to understand whether scaling is required as a preprocessing step.

6.  Import the `StandardScaler` class from scikit-learn\'s
    preprocessing package. Initialize the scaler, fit the scaler, and
    transform the DataFrame using the scaler\'s
    `fit_transform` method. Create a new DataFrame,
    `df2`, using the transformed DataFrame since the result of
    the `fit_transform` method is a NumPy array. The standard
    scaler will transform the numerical fields so that the mean of the
    field is `0` and the standard deviation is `1`:

    
    ```
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df2 = scaler.fit_transform(df)
    df2 = pd.DataFrame(df2, columns=df.columns)
    ```


    Note

    The values for the mean and standard deviation of the resulting
    transformed data can be input into the scaler.

7.  Plot a histogram of the transformed `Present_Tmax` column:

    
    ```
    ax = df2['Present_Tmax'].hist(color='gray')
    ax.set_xlabel("Normalized Temperature")
    ax.set_ylabel("Frequency")
    ```


    The output will be as follows:

    
![](./images/B16341_02_07.jpg)


The resulting histogram shows that the temperature values range from
around `-3` to `3` degrees Celsius, as evidenced by
the range on the *x* axis of the histogram. By using the standard
scaler, the values will always have a mean of `0` and a
standard deviation of `1`. Having the features normalized can
speed up the model training process.

In this exercise, you successfully imported tabular data using the
pandas library and performed some preprocessing using the scikit-learn
library. The preprocessing of data included dropping the
`date` column and scaling the numerical fields so that they
have a mean value of `0` and a standard deviation of
`1`.

In the following activity, you will load in tabular data using the
pandas library and scale that data using the `MinMax` scaler
present in scikit-learn. You will do so on the same dataset that you
used in the prior exercise, which describes the bias correction of air
temperature forecasts for Seoul, South Korea.

Activity 2.01: Loading Tabular Data and Rescaling Numerical Fields with a MinMax Scaler
---------------------------------------------------------------------------------------

In this activity, you are required to load tabular data and rescale the
data using a `MinMax` scaler. The dataset,
`Bias_correction_ucl.csv`, contains information for bias
correction of the next-day maximum and minimum air temperature forecast
for Seoul, South Korea. The fields represent temperature measurements of
the given date, the weather station at which the metrics were measured,
model forecasts of weather-related metrics such as humidity, and
projections for the temperature the following day. You are required to
scale the columns so that the minimum value of each column is
`0` and the maximum value is `1`.

Perform the following steps to complete this activity:

1.  Open a new Jupyter notebook to implement this activity.

2.  Import pandas and the `Bias_correction_ucl.csv` dataset.

3.  Read the dataset using the pandas `read_csv` function.

4.  Drop the `date` column of the DataFrame.

5.  Plot a histogram of the `Present_Tmax` column.

6.  Import `MinMaxScaler` and fit it to and transform the
    feature DataFrame.

7.  Plot a histogram of the transformed `Present_Tmax` column.

    You should get an output similar to the following:

    
![](./images/B16341_02_08.jpg)




One method of converting non-numerical fields such as categorical or
date fields is to one-hot encode them. The **one-hot encoding process**
creates a new column for each unique value in the provided column, while
each row has a value of `0` except for the one that
corresponds to the correct column. The column headers of the newly
created dummy columns correspond to the unique values. One-hot encoding
can be achieved by using the `get_dummies` function of the
pandas library and passing in the column to be encoded. An optional
argument is to provide a prefix feature that adds a prefix to the column
headers. This can be useful for referencing the columns:


```
dummies = pd.get_dummies(df['feature1'], prefix='feature1')
```


Note

When using the `get_dummies` function, `NaN` values
are converted into all zeros.

In the following exercise, you\'ll learn how to preprocess non-numerical
fields. You will utilize the same dataset that you used in the previous
exercise and activity, which describes the bias correction of air
temperature forecasts for Seoul, South Korea.

Exercise 2.02: Preprocessing Non-Numerical Data
-----------------------------------------------

In this exercise, you will preprocess the `date` column by
one-hot encoding the year and the month from the `date` column
using the `get_dummies` function. You will join the
one-hot-encoded columns with the original DataFrame and ensure that all
the fields in the resultant DataFrame are numerical.

Perform the following steps to complete this exercise:

1.  Open a new Jupyter notebook to implement this exercise. Save the
    file as `Exercise2-02.ipnyb`.

2.  In a new Jupyter Notebook cell, import the pandas library, as
    follows:
    
    ```
    import pandas as pd
    ```


3.  Create a new pandas DataFrame named `df` and read the
    `Bias_correction_ucl.csv` file into it. Examine whether
    your data is properly loaded by printing the resultant DataFrame:

    
    ```
    df = pd.read_csv('Bias_correction_ucl.csv')
    ```


    Note

    Make sure you change the path (highlighted) to the CSV file based on
    its location on your system. If you\'re running the Jupyter notebook
    from the same directory where the CSV file is stored, you can run
    the preceding code without any modification.

4.  Change the data type of the `date` column to
    `Date` using the pandas `to_datetime` function:
    
    ```
    df['Date'] = pd.to_datetime(df['Date'])
    ```


5.  Create dummy columns for `year` using the pandas
    `get_dummies` function. Pass in the year of the
    `date` column as the first argument and add a prefix to
    the columns of the resultant DataFrame. Print out the resultant
    DataFrame:

    
    ```
    year_dummies = pd.get_dummies(df['Date'].dt.year, \
                                  prefix='year')
    year_dummies
    ```


    The output will be as follows:

    
![](./images/B16341_02_09.jpg)



    The resultant DataFrame contains only 0s and 1s. `1`
    corresponds to the value present in the original `date`
    column. Null values will have 0s for all columns in the newly
    created DataFrame.

6.  Repeat this for the month by creating dummy columns from the month
    of the `date` column. Print out the resulting DataFrame:

    
    ```
    month_dummies = pd.get_dummies(df['Date'].dt.month, \
                                   prefix='month')
    month_dummies
    ```


    The output will be as follows:

    
![](./images/B16341_02_10.jpg)



    The resultant DataFrame now contains only 0s and 1s for the month in
    the `date` column.

7.  Concatenate the original DataFrame and the dummy DataFrames you
    created in *Steps 5* and *6*:
    
    ```
    df = pd.concat([df, month_dummies, year_dummies], \
                   axis=1)
    ```


8.  Drop the original `date` column since it is now redundant:
    
    ```
    df.drop('Date', axis=1, inplace=True)
    ```


9.  Verify that all the columns are now of the numerical data type:

    
    ```
    df.dtypes
    ```


    The output will be as follows:

    
![](./images/B16341_02_11.jpg)




Here, you can see that all the data types of the resultant DataFrame are
numerical. This means they can now be passed into an ANN for modeling.

In this exercise, you successfully imported tabular data and
preprocessed the `date` column using the pandas and
scikit-learn libraries. You utilized the `get_dummies`
function to convert categorical data into numerical data types.



In the next section, you will learn how to process image data so that it
can be input into machine learning models.


Procesing Image Data
=====================


In the following exercise, you will load images into memory by utilizing
the `ImageDataGenerator` class.

Note

The image data provided comes from the Open Image dataset, a full
description of which can be found here:
[https://storage.googleapis.com/openimages/web/index.html].


Images can be viewed by plotting them using Matplotlib. This is a useful
exercise for verifying that the images match their respective labels.

Exercise 2.03: Loading Image Data for Batch Processing
------------------------------------------------------

In this exercise, you\'ll learn how to load in image data for batch
processing. The `image_data` folder contains a set of images
of boats and airplanes. You will load the images of boats and airplanes
for batch processing and rescale them so that the image values range
between `0` and `1`. You are then tasked with
printing the labeled images of a batch from the data generator.

Note

You can find `image_data` here:
[https://github.com/fenago/deep-learning-essentials/tree/main/Lab02/Datasets].

Perform the following steps to complete this exercise:

1.  Open a new Jupyter notebook to implement this exercise. Save the
    file as `Exercise2-03.ipnyb`.

2.  In a new Jupyter Notebook cell, import the
    `ImageDataGenerator` class from
    `tensorflow.keras.preprocessing.image`:
    
    ```
    from tensorflow.keras.preprocessing.image \
         import ImageDataGenerator
    ```


3.  Instantiate the `ImageDataGenerator` class and pass the
    `rescale` argument with the value `1./255` to
    convert image values so that they\'re between `0` and
    `1`:
    
    ```
    train_datagen = ImageDataGenerator(rescale =  1./255)
    ```


4.  Use the data generator\'s `flow_from_directory` method to
    direct the data generator to the image data. Pass in the arguments
    for the target size, the batch size, and the class mode:
    
    ```
    training_set = train_datagen.flow_from_directory\
                   ('image_data',\
                    target_size = (64, 64),\
                    batch_size = 25,\
                    class_mode = 'binary')
    ```


5.  Create a function to display the images in the batch. The function
    will plot the first 25 images in a 5x5 array with their associated
    labels:
    
    ```
    import matplotlib.pyplot as plt
    def show_batch(image_batch, label_batch):\
        lookup = {v: k for k, v in \
                  training_set.class_indices.items()}
        label_batch = [lookup[label] for label in \
                       label_batch]
        plt.figure(figsize=(10,10))
        for n in range(25):
            ax = plt.subplot(5,5,n+1)
            plt.imshow(image_batch[n])
            plt.title(label_batch[n].title())
            plt.axis('off')
    ```


6.  Take a batch from the data generator and pass it to the function to
    display the images and their labels:

    
    ```
    image_batch, label_batch = next(training_set)
    show_batch(image_batch, label_batch)
    ```


    The output will be as follows:

    
![](./images/B16341_02_12.jpg)




Here, you can see the output of a batch of images of boats and airplanes
that can be input into a model. Note that all the images are the same
size, which was achieved by modifying the aspect ratio of the images.
This ensures consistency in the images as they are passed into an ANN.

In this exercise, you learned how to import images in batches so they
can be used for training ANNs. Images are loaded one batch at a time and
by limiting the number of training images per batch, you can ensure that
the RAM of the machine is not exceeded.

In the following section, you will see how to augment images as they are
loaded in.





Image Augmentation
==================

In the following activity, you perform image augmentation using
TensorFlow\'s `ImageDataGenerator` class. The process is as
simple as passing in parameters. You will use the same dataset that you
used in *Exercise 2.03*, *Loading Image Data for Batch Processing*,
which contains images of boats and airplanes.

Activity 2.02: Loading Image Data for Batch Processing
------------------------------------------------------

In this activity, you will load image data for batch processing and
augment the images in the process. The `image_data` folder
contains a set of images of boats and airplanes. You are required to
load in image data for batch processing and adjust the input data with
random perturbations such as rotations, flipping the image horizontally,
and adding shear to the images. This will create additional training
data from the existing image data and will lead to more accurate and
robust machine learning models by increasing the number of different
training examples even if only a few are available. You are then tasked
with printing the labeled images of a batch from the data generator.

The steps for this activity are as follows:

1.  Open a new Jupyter notebook to implement this activity.

2.  Import the `ImageDataGenerator` class from
    `tensorflow.keras.preprocessing.image`.

3.  Instantiate `ImageDataGenerator` and set the
    `rescale=1./255`, `shear_range=0.2`,
    `rotation_range=180`, `zoom_range=0.2`,
    and `horizontal_flip=True` arguments.

4.  Use the `flow_from_directory` method to direct the data
    generator to the images while passing in the target size as
    `64x64`, a batch size of `25`, and the class
    mode as `binary`.

5.  Create a function to display the first 25 images in a 5x5 array with
    their associated labels.

6.  Take a batch from the data generator and pass it to the function to
    display the images and their labels.



In this activity, you augmented images in batches so they could be used
for training ANNs. You\'ve seen that when images are used as input, they
can be augmented to generate a larger number of effective training
examples.

You learned how to load images in batches, which enables you to train on
huge volumes of data that may not fit into the memory of your machine at
one time. You also learned how to augment images using the
`ImageDataGenerator` class, which essentially generates new
training examples from the images in your training set.

In the next section, you will learn how to load and preprocess text
data.



Text Processing
===============


Text data represents a large class of raw data that is readily
available. For example, text data can be from web pages such as
Wikipedia, transcribed speech, or social media conversations---all of
which are increasing at a massive scale and must be processed before
they can be used for training machine learning models.


In the following exercise, you will explore how to load in data that
includes a text field, batch the dataset, and apply a pretrained model
to the text field to convert the field into embedded vectors.

Note

The pretrained model can be found here:
[https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1].

The dataset can be found here:
[https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29].

Exercise 2.04: Loading Text Data for TensorFlow Models
------------------------------------------------------

The dataset, `drugsComTrain_raw.tsv`, contains information
related to patient reviews on specific drugs, along with their related
conditions and a rating indicating the patient\'s satisfaction with the
drug. In this exercise, you will load in text data for batch processing.
You will apply a pretrained model from TensorFlow Hub to perform a word
embedding on the patient reviews. You are required to work on the
`review` field only as that contains text data.

Perform the following steps:

1.  Open a new Jupyter notebook to implement this exercise. Save the
    file as `Exercise2-04.ipnyb`.

2.  In a new Jupyter Notebook cell, import the TensorFlow library:
    
    ```
    import tensorflow as tf
    ```


3.  Create a TensorFlow dataset object using the library\'s
    `make_csv_dataset` function. Set the
    `batch_size` argument equal to `1` and the
    `field_delim` argument to `'\t'` since the
    dataset is tab-delimited:
    
    ```
    df = tf.data.experimental.make_csv_dataset\
         ('../Datasets/drugsComTest_raw.tsv', \
          batch_size=1, field_delim='\t')
    ```


4.  Create a function that takes a dataset object as input and shuffles,
    repeats, and batches the dataset:
    
    ```
    def prep_ds(ds, shuffle_buffer_size=1024, \
                batch_size=32):
        # Shuffle the dataset
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat the dataset
        ds = ds.repeat()
        # Batch the dataset
        ds = ds.batch(batch_size)
        return ds
    ```


5.  Apply the function to the dataset object you created in *Step 3*,
    setting `batch_size` equal to `5`:
    
    ```
    ds = prep_ds(df, batch_size=5)
    ```


6.  Take the first batch and print it out:

    
    ```
    for x in ds.take(1):\
        print(x)
    ```


    You should get output similar to the following:

    
![](./images/B16341_02_14.jpg)




    The output represents the input data in tensor format.

7.  Import the pretrained word embedding model from TensorFlow Hub and
    create a Keras layer:
    
    ```
    import tensorflow_hub as hub
    embedding = "https://tfhub.dev/google/tf2-preview"\
                "/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[], \
                               dtype=tf.string, \
                               trainable=True)
    ```


8.  Take one batch from the dataset, flatten the tensor corresponding to
    the `review` field, apply the pretrained layer, and print
    it out:

    
    ```
    for x in ds.take(1):\
        print(hub_layer(tf.reshape(x['review'],[-1])))
    ```


    This will display the following output:

    
![](./images/B16341_02_15.jpg)




The preceding output represents the embedding vectors for the first
batch of drug reviews. The specific values may not mean much at first
glance but encoded within the embeddings is contextual information based
on the dataset that the embedding model was trained upon. The batch size
is equal to `5` and the embedding vector size is
`20`, which means the resulting size, after applying the
pretrained layer, is `5x20`.



In the next section, you will learn how to load and process audio data
so that the data can be used for TensorFlow models.



Audio Processing
================

In the following exercise, you will understand how audio data can be
processed. In a similar manner to what you did in *Exercise 2.03*,
*Loading Image Data for Batch Processing*, and *Exercise* *2.04*,
*Loading Text Data for TensorFlow Models*, you will load the data in
batches for efficient and scalable training. You will load in the audio
files using TensorFlow\'s generic `read_file` function, then
decode the audio data using TensorFlow\'s `decode_wav`
function. You will then create a function that will generate the MFCCs
from each audio sample. Finally, a dataset object will be generated that
can be passed into a TensorFlow model for training. The dataset that you
will be utilizing is Google\'s speech commands dataset, which consists
of 1-second-long utterances of words.

Note

The dataset can be found here: [https://github.com/fenago/deep-learning-essentials/tree/main/Lab02/Datasets/data_speech_commands_v0.02].

Exercise 2.05: Loading Audio Data for TensorFlow Models
-------------------------------------------------------

In this exercise, you\'ll learn how to load in audio data for batch
processing. The dataset, `data_speech_commands_v0.02`,
contains speech samples of people speaking the word `zero` for
exactly 1 second with a sample rate of 44.1 kHz, meaning that for every
second, there are 44,100 data points. You will apply some common audio
preprocessing techniques, including converting the data into the Fourier
domain, sampling the data to ensure the data has the same size as the
model, and generating MFCCs for each audio sample. This will generate a
preprocessed dataset object that can be input into a TensorFlow model
for training.

Perform the following steps:

1.  Open a new Jupyter notebook to implement this exercise. Save the
    file as `Exercise2-05.ipnyb`.

2.  In a new Jupyter Notebook cell, import the `tensorflow`
    and `os` libraries:
    
    ```
    import tensorflow as tf
    import os
    ```


3.  Create a function that will load an audio file using TensorFlow\'s
    `read_file` function and `decode_wav` function,
    respectively. Return the transpose of the resultant tensor:
    
    ```
    def load_audio(file_path, sample_rate=44100):
        # Load audio at 44.1kHz sample-rate
        audio = tf.io.read_file(file_path)
        audio, sample_rate = tf.audio.decode_wav\
                             (audio,\
                              desired_channels=-1,\
                              desired_samples=sample_rate)
        return tf.transpose(audio)
    ```


4.  Load in the paths to the audio data as a list using
    `os.list_dir`:
    
    ```
    prefix = " ../Datasets/data_speech_commands_v0.02"\
            "/zero/"
    paths = [os.path.join(prefix, path) for path in \
             os.listdir(prefix)]
    ```


5.  Test the function by loading in the first audio file from the list
    and plotting it:

    
    ```
    import matplotlib.pyplot as plt
    audio = load_audio(paths[0])
    plt.plot(audio.numpy().T)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    ```


    The output will be as follows:

    
![](./images/B16341_02_16.jpg)




    The figure shows the waveform of the speech sample. The amplitude at
    a given time corresponds to the volume of the sound; high amplitude
    relates to high volume.

6.  Create a function to generate the MFCCs from the audio data. First,
    apply the short-time Fourier transform passing in the audio signal
    as the first argument, the frame length set to `1024` as
    the second argument, the frame step set to `256` as the
    third argument, and the FFT length as the fourth parameter. Then,
    take the absolute value of the result to compute the spectrograms.
    The number of spectrogram bins is given by the length along the last
    axis of the short-time Fourier transform. Next, define the upper and
    lower bounds of the mel weight matrix as `80` and
    `7600` respectively and the number of mel bins as
    `80`. Then, compute the mel weight matrix using
    `linear_to_mel_weight_matrix` from TensorFlow\'s signal
    package. Next, compute the mel spectrograms via tensor contraction
    using TensorFlow\'s `tensordot` function along axis 1 of
    the spectrograms with the mel weight matrix. Then, take the log of
    the mel spectrograms before finally computing the MFCCs using
    TensorFlow\'s `mfccs_from_log_mel_spectrograms` function.
    Then, return the MFCCs from the function:
    
    ```
    def apply_mfccs(audio, sample_rate=44100, num_mfccs=13):
        stfts = tf.signal.stft(audio, frame_length=1024, \
                               frame_step=256, \
                               fft_length=1024)
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]#.value
        lower_edge_hertz, upper_edge_hertz, \
        num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = \
          tf.signal.linear_to_mel_weight_matrix\
          (num_mel_bins, num_spectrogram_bins, \
           sample_rate, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot\
                           (spectrograms, \
                            linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape\
        (spectrograms.shape[:-1].concatenate\
        (linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.math.log\
                               (mel_spectrograms + 1e-6)
        #Compute MFCCs from log_mel_spectrograms
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms\
                (log_mel_spectrograms)[..., :num_mfccs]
        return mfccs
    ```


7.  Apply the function to generate the MFCCs for the audio data you
    loaded in *Step 5*:

    
    ```
    mfcc = apply_mfccs(audio)
    plt.pcolor(mfcc.numpy()[0])
    plt.xlabel('MFCC log coefficient')
    plt.ylabel('Sample Value')
    ```


    The output will be as follows:

    
![](./images/B16341_02_17.jpg)




    The preceding plot shows the MFCC values on the *x* axis and various
    points of the audio sample on the *y* axis. MFCCs are a different
    representation of the raw audio signal displayed in *Step 5* that
    has been proven to be useful in applications related to speech
    recognition.

8.  Load `AUTOTUNE` so that you can use all the available
    threads of the CPU. Create a function that will take a dataset
    object, shuffle it, load the audio using the function you created in
    *Step 3*, generate the MFCCs using the function you created in *Step
    6*, repeat the dataset object, batch it, and prefetch it. Use
    `AUTOTUNE` to prefetch with a buffer size based on your
    available CPU:
    
    ```
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    def prep_ds(ds, shuffle_buffer_size=1024, \
                batch_size=64):
        # Randomly shuffle (file_path, label) dataset
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Load and decode audio from file paths
        ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)
        # generate MFCCs from the audio data
        ds = ds.map(apply_mfccs)
        # Repeat dataset forever
        ds = ds.repeat()
        # Prepare batches
        ds = ds.batch(batch_size)
        # Prefetch
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    ```


9.  Generate the training dataset using the function you created in
    *Step 8*. To do this, create a dataset object using TensorFlow\'s
    `from_tensor_slices` function and pass in the paths to the
    audio files. After that, you can use the function you created in
    *Step 8*:
    
    ```
    ds = tf.data.Dataset.from_tensor_slices(paths)
    train_ds = prep_ds(ds)
    ```


10. Take the first batch of the dataset and print it out:

    
    ```
    for x in train_ds.take(1):\
        print(x)
    ```


    The output will be as follows:

    
![](./images/B16341_02_18.jpg)



The output shows the first batch of MFCC spectrum values in tensor form.

In this exercise, you imported audio data. You processed the dataset and
batched the dataset so that it is appropriate for large-scale training.
This method was a comprehensive approach in which the data was loaded
and converted into the frequency domain, spectrograms were generated,
and then finally the MFCCs were generated.

In the next activity, you will load in audio data and take the absolute
value of the input, followed by scaling the values logarithmically. This
will ensure that there are no negative values in the dataset. You will
use the same audio dataset that you used in *Exercise 2.05*, *Loading
Audio Data for TensorFlow Models*, that is, Google\'s speech commands
dataset. This dataset consists of 1-second-long utterances of words.

Activity 2.03: Loading Audio Data for Batch Processing
------------------------------------------------------

In this activity, you will load audio data for batch processing. The
audio preprocessing techniques that will be performed include taking the
absolute value and using the logarithm of 1 plus the value. This will
ensure the resulting values are non-negative and logarithmically scaled.
The result will be a preprocessed dataset object that can be input into
a TensorFlow model for training.

The steps for this activity are as follows:

1.  Open a new Jupyter notebook to implement this activity.

2.  Import the TensorFlow and `os` libraries.

3.  Create a function that will load and then decode an audio file using
    TensorFlow\'s `read_file` function followed by the
    `decode_wav` function, respectively. Return the transpose
    of the resultant tensor from the function.

4.  Load the file paths into the audio data as a list using
    `os.list_dir`.

5.  Create a function that takes a dataset object, shuffles it, loads
    the audio using the function you created in *step 2*, and applies
    the absolute value and the `log1p` function to the
    dataset. This function adds `1` to each value in the
    dataset and then applies the logarithm to the result. Next, repeat
    the dataset object, batch it, and prefetch it with a buffer size
    equal to the batch size.

6.  Create a dataset object using TensorFlow\'s
    `from_tensor_slices` function and pass in the paths to the
    audio files. Then, apply the function you created in *Step 4* to the
    dataset created in *Step 5*.

7.  Take the first batch of the dataset and print it out.

8.  Plot the first audio file from the batch.

    The output will look as follows:

    
![](./images/B16341_02_19.jpg)




In this activity, you learned how to load and preprocess audio data in
batches. You used most of the functions that you used in *Exercise
2.05*, *Loading Audio Data for TensorFlow Models*, to load in the data
and decode the raw data. The difference between *Exercise 2.05*,
*Loading Audio Data for TensorFlow Models*, and *Activity 2.03*,
*Loading Audio Data for Batch Processing*, is the preprocessing steps;
*Exercise 2.05*, *Loading Audio Data for TensorFlow Models*, involved
generating MFCCs for the audio data, whereas *Activity 2.03*, *Loading
Audio Data for Batch Processing*, involved scaling the data
logarithmically. Both demonstrate common preprocessing techniques that
can be used for all applications involving modeling on audio data.



Summary
=======

In this lab:

1. Loaded and preprocessed tabular data from a CSV file using pandas, scaling and converting fields to numerical types for compatibility with TensorFlow models.
2. Batched and augmented image data, enhancing training with more robust datasets.
3. Embedded text data into numerical vectors using pretrained models, making it suitable for TensorFlow inputs.
4. Preprocessed audio data, generating MFCCs to create dense numerical tensors for TensorFlow models.
5. Highlighted the importance of data preprocessing for efficient and accurate machine learning model training.
