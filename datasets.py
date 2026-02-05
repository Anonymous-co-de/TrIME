import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def reduce_mem_usage(df, verbose=True):
  """Function to reduce un-needed memory.

  Args:
    df: pandas dataframe to reduce size
    verbose: bool indicating if whether to print size reduction

  Returns:
    df: new pandas dataframe  with reduced size
  """

  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  start_mem = df.memory_usage().sum() / 1024**2
  for col in df.columns:
    col_type = df[col].dtypes
    if col_type in numerics:
      c_min = df[col].min()
      c_max = df[col].max()
      if str(col_type)[:3] == 'int':
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
          df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
          df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
          df[col] = df[col].astype(np.int32)
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
          df[col] = df[col].astype(np.int64)
      else:
        if c_min > np.finfo(np.float16).min and c_max < np.finfo(
            np.float16).max:
          df[col] = df[col].astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
            np.float32).max:
          df[col] = df[col].astype(np.float32)
        else:
          df[col] = df[col].astype(np.float64)
  end_mem = df.memory_usage().sum() / 1024**2
  if verbose:
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
        end_mem, 100 * (start_mem - end_mem) / start_mem))
  return df

class ECL(Dataset):
  """Create an electricity dataset."""

  def __init__(self,
               root_path,
               seq_len,
               pred_len,
               features='S',
               scale=True,
               num_ts=30):
    """Initializes a ECL instance.

    Args:
     root_path: root path of the data file
     seq_len: input sequence length'
     pred_len: prediction sequence length
     features: forecasting task, options:[S, MS]; S:univariate predict
       univariate, MS:multivariate predict univariate
     scale: bool indicating whether to scale the dataset
     num_ts: Number Of Independent Time Series to Model
    """
    self.root_path = root_path
    self.seq_len = seq_len
    self.pred_len = pred_len

    self.features = features
    self.scale = scale

    self.num_ts = num_ts + 1
    self.__read_data__()

  def __read_data__(self):

    # Standard scaler is used to scale data
    if self.scale:
      self.scaler = StandardScaler()
    # Loading dataset
    data_path = os.path.join(self.root_path, 'ECL.csv')
    inputs = pd.read_csv(data_path)
    inputs.ffill(inplace=True)
    inputs = reduce_mem_usage(inputs)

    # Dividing dataset into Training, Testing and Validation
    total_number_of_hours = inputs.shape[0]
    training_hours = int(inputs.shape[0] * 0.7)
    testing_hours = int(inputs.shape[0] * 0.2)
    validation_hours = total_number_of_hours - training_hours - testing_hours

    train_x = []
    train_y = []
    train_index = []

    valid_x = []
    valid_y = []
    valid_index = []

    test_x = []
    test_y = []
    test_index = []

    if self.features == 'S':
      train_sample_index = 0
      test_sample_index = 0
      val_sample_index = 0

      # Every customer (an entire time series) divide the customer data
      # into training/testing/validation,
      # Scale indpendently since customers have very different behaviors
      # Create time series based on the sequence and prediction length
      for i in range(1, self.num_ts):

        data = inputs.values[:, i]
        data = data.reshape(-1, 1)
        if self.scale:
          train_data = np.array(data[:training_hours, :], dtype=np.float64)
          self.scaler.fit(train_data)
          df_data = self.scaler.transform(data)
        else:
          df_data = np.array(data, dtype=np.float64)

        index = 0
        for j in range(index, training_hours - self.pred_len - self.seq_len):

          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1
          train_x.append(df_data[s_begin:s_end, :])
          train_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          train_index.append(train_sample_index)
          train_sample_index += 1

        index = training_hours - self.pred_len - self.seq_len
        for j in range(
            index,
            validation_hours + training_hours - self.pred_len - self.seq_len):

          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1

          valid_x.append(df_data[s_begin:s_end, :])
          valid_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          valid_index.append(val_sample_index)
          val_sample_index += 1

        index = validation_hours + training_hours - self.pred_len - self.seq_len
        for j in range(
            index, testing_hours + validation_hours + training_hours -
            self.pred_len - self.seq_len):
          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1
          test_x.append(df_data[s_begin:s_end, :])
          test_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          test_index.append(test_sample_index)
          test_sample_index += 1
    else:
      # Electricity is a univariate dataset
      raise NotImplementedError

    self.train_x = np.array(train_x, dtype=np.float16)
    self.train_y = np.array(train_y, dtype=np.float16)
    self.train_index = np.array(train_index)

    self.valid_x = np.array(valid_x, dtype=np.float16)
    self.valid_y = np.array(valid_y, dtype=np.float16)
    self.valid_index = np.array(valid_index)

    self.test_x = np.array(test_x, dtype=np.float16)
    self.test_y = np.array(test_y, dtype=np.float16)
    self.test_index = np.array(test_index)

class Climate(Dataset):
  """Create an electricity dataset."""

  def __init__(self,
               root_path,
               seq_len,
               pred_len,
               features='S',
               scale=True,
               num_ts=30):
    """Initializes a ECL instance.

    Args:
     root_path: root path of the data file
     seq_len: input sequence length'
     pred_len: prediction sequence length
     features: forecasting task, options:[S, MS]; S:univariate predict
       univariate, MS:multivariate predict univariate
     scale: bool indicating whether to scale the dataset
     num_ts: Number Of Independent Time Series to Model
    """
    self.root_path = root_path
    self.seq_len = seq_len
    self.pred_len = pred_len

    self.features = features
    self.scale = scale

    self.num_ts = num_ts + 1
    self.__read_data__()

  def __read_data__(self):

    # Standard scaler is used to scale data
    if self.scale:
      self.scaler = StandardScaler()
    # Loading dataset
    data_path = os.path.join(self.root_path, 'Climate.csv')
    inputs = pd.read_csv(data_path)
    inputs.ffill(inplace=True)
    inputs = reduce_mem_usage(inputs)

    # Dividing dataset into Training, Testing and Validation
    total_number_of_hours = inputs.shape[0]
    training_hours = int(inputs.shape[0] * 0.7)
    testing_hours = int(inputs.shape[0] * 0.2)
    validation_hours = total_number_of_hours - training_hours - testing_hours

    train_x = []
    train_y = []
    train_index = []

    valid_x = []
    valid_y = []
    valid_index = []

    test_x = []
    test_y = []
    test_index = []

    if self.features == 'S':
      train_sample_index = 0
      test_sample_index = 0
      val_sample_index = 0

      # Every customer (an entire time series) divide the customer data
      # into training/testing/validation,
      # Scale indpendently since customers have very different behaviors
      # Create time series based on the sequence and prediction length
      for i in range(1, self.num_ts):

        data = inputs.values[:, i]
        data = data.reshape(-1, 1)
        if self.scale:
          train_data = np.array(data[:training_hours, :], dtype=np.float64)
          self.scaler.fit(train_data)
          df_data = self.scaler.transform(data)
        else:
          df_data = np.array(data, dtype=np.float64)

        index = 0
        for j in range(index, training_hours - self.pred_len - self.seq_len):

          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1
          train_x.append(df_data[s_begin:s_end, :])
          train_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          train_index.append(train_sample_index)
          train_sample_index += 1

        index = training_hours - self.pred_len - self.seq_len
        for j in range(
            index,
            validation_hours + training_hours - self.pred_len - self.seq_len):

          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1

          valid_x.append(df_data[s_begin:s_end, :])
          valid_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          valid_index.append(val_sample_index)
          val_sample_index += 1

        index = validation_hours + training_hours - self.pred_len - self.seq_len
        for j in range(
            index, testing_hours + validation_hours + training_hours -
            self.pred_len - self.seq_len):
          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1
          test_x.append(df_data[s_begin:s_end, :])
          test_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          test_index.append(test_sample_index)
          test_sample_index += 1
    else:
      # Electricity is a univariate dataset
      raise NotImplementedError

    self.train_x = np.array(train_x, dtype=np.float16)
    self.train_y = np.array(train_y, dtype=np.float16)
    self.train_index = np.array(train_index)

    self.valid_x = np.array(valid_x, dtype=np.float16)
    self.valid_y = np.array(valid_y, dtype=np.float16)
    self.valid_index = np.array(valid_index)

    self.test_x = np.array(test_x, dtype=np.float16)
    self.test_y = np.array(test_y, dtype=np.float16)
    self.test_index = np.array(test_index)

class ETTh1(Dataset):
  """Create an electricity dataset."""

  def __init__(self,
               root_path,
               seq_len,
               pred_len,
               features='S',
               scale=True,
               num_ts=30):
    """Initializes a ECL instance.

    Args:
     root_path: root path of the data file
     seq_len: input sequence length'
     pred_len: prediction sequence length
     features: forecasting task, options:[S, MS]; S:univariate predict
       univariate, MS:multivariate predict univariate
     scale: bool indicating whether to scale the dataset
     num_ts: Number Of Independent Time Series to Model
    """
    self.root_path = root_path
    self.seq_len = seq_len
    self.pred_len = pred_len

    self.features = features
    self.scale = scale

    self.num_ts = num_ts + 1
    self.__read_data__()

  def __read_data__(self):

    # Standard scaler is used to scale data
    if self.scale:
      self.scaler = StandardScaler()
    # Loading dataset
    data_path = os.path.join(self.root_path, 'ETTh1.csv')
    inputs = pd.read_csv(data_path)
    inputs.ffill(inplace=True)
    inputs = reduce_mem_usage(inputs)

    # Dividing dataset into Training, Testing and Validation
    total_number_of_hours = inputs.shape[0]
    training_hours = int(inputs.shape[0] * 0.7)
    testing_hours = int(inputs.shape[0] * 0.2)
    validation_hours = total_number_of_hours - training_hours - testing_hours

    train_x = []
    train_y = []
    train_index = []

    valid_x = []
    valid_y = []
    valid_index = []

    test_x = []
    test_y = []
    test_index = []

    if self.features == 'S':
      train_sample_index = 0
      test_sample_index = 0
      val_sample_index = 0

      # Every customer (an entire time series) divide the customer data
      # into training/testing/validation,
      # Scale indpendently since customers have very different behaviors
      # Create time series based on the sequence and prediction length
      for i in range(1, self.num_ts):

        data = inputs.values[:, i]
        data = data.reshape(-1, 1)
        if self.scale:
          train_data = np.array(data[:training_hours, :], dtype=np.float64)
          self.scaler.fit(train_data)
          df_data = self.scaler.transform(data)
        else:
          df_data = np.array(data, dtype=np.float64)

        index = 0
        for j in range(index, training_hours - self.pred_len - self.seq_len):

          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1
          train_x.append(df_data[s_begin:s_end, :])
          train_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          train_index.append(train_sample_index)
          train_sample_index += 1

        index = training_hours - self.pred_len - self.seq_len
        for j in range(
            index,
            validation_hours + training_hours - self.pred_len - self.seq_len):

          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1

          valid_x.append(df_data[s_begin:s_end, :])
          valid_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          valid_index.append(val_sample_index)
          val_sample_index += 1

        index = validation_hours + training_hours - self.pred_len - self.seq_len
        for j in range(
            index, testing_hours + validation_hours + training_hours -
            self.pred_len - self.seq_len):
          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1
          test_x.append(df_data[s_begin:s_end, :])
          test_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          test_index.append(test_sample_index)
          test_sample_index += 1
    else:
      # Electricity is a univariate dataset
      raise NotImplementedError

    self.train_x = np.array(train_x, dtype=np.float16)
    self.train_y = np.array(train_y, dtype=np.float16)
    self.train_index = np.array(train_index)

    self.valid_x = np.array(valid_x, dtype=np.float16)
    self.valid_y = np.array(valid_y, dtype=np.float16)
    self.valid_index = np.array(valid_index)

    self.test_x = np.array(test_x, dtype=np.float16)
    self.test_y = np.array(test_y, dtype=np.float16)
    self.test_index = np.array(test_index)

class ETTh2(Dataset):
  """Create an electricity dataset."""

  def __init__(self,
               root_path,
               seq_len,
               pred_len,
               features='S',
               scale=True,
               num_ts=30):
    """Initializes a ECL instance.

    Args:
     root_path: root path of the data file
     seq_len: input sequence length'
     pred_len: prediction sequence length
     features: forecasting task, options:[S, MS]; S:univariate predict
       univariate, MS:multivariate predict univariate
     scale: bool indicating whether to scale the dataset
     num_ts: Number Of Independent Time Series to Model
    """
    self.root_path = root_path
    self.seq_len = seq_len
    self.pred_len = pred_len

    self.features = features
    self.scale = scale

    self.num_ts = num_ts + 1
    self.__read_data__()

  def __read_data__(self):

    # Standard scaler is used to scale data
    if self.scale:
      self.scaler = StandardScaler()
    # Loading dataset
    data_path = os.path.join(self.root_path, 'ETTh2.csv')
    inputs = pd.read_csv(data_path)
    inputs.ffill(inplace=True)
    inputs = reduce_mem_usage(inputs)

    # Dividing dataset into Training, Testing and Validation
    total_number_of_hours = inputs.shape[0]
    training_hours = int(inputs.shape[0] * 0.7)
    testing_hours = int(inputs.shape[0] * 0.2)
    validation_hours = total_number_of_hours - training_hours - testing_hours

    train_x = []
    train_y = []
    train_index = []

    valid_x = []
    valid_y = []
    valid_index = []

    test_x = []
    test_y = []
    test_index = []

    if self.features == 'S':
      train_sample_index = 0
      test_sample_index = 0
      val_sample_index = 0

      # Every customer (an entire time series) divide the customer data
      # into training/testing/validation,
      # Scale indpendently since customers have very different behaviors
      # Create time series based on the sequence and prediction length
      for i in range(1, self.num_ts):

        data = inputs.values[:, i]
        data = data.reshape(-1, 1)
        if self.scale:
          train_data = np.array(data[:training_hours, :], dtype=np.float64)
          self.scaler.fit(train_data)
          df_data = self.scaler.transform(data)
        else:
          df_data = np.array(data, dtype=np.float64)

        index = 0
        for j in range(index, training_hours - self.pred_len - self.seq_len):

          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1
          train_x.append(df_data[s_begin:s_end, :])
          train_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          train_index.append(train_sample_index)
          train_sample_index += 1

        index = training_hours - self.pred_len - self.seq_len
        for j in range(
            index,
            validation_hours + training_hours - self.pred_len - self.seq_len):

          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1

          valid_x.append(df_data[s_begin:s_end, :])
          valid_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          valid_index.append(val_sample_index)
          val_sample_index += 1

        index = validation_hours + training_hours - self.pred_len - self.seq_len
        for j in range(
            index, testing_hours + validation_hours + training_hours -
            self.pred_len - self.seq_len):
          s_begin = j
          s_end = s_begin + self.seq_len
          r_begin = s_end + self.pred_len - 1
          r_end = r_begin + 1
          test_x.append(df_data[s_begin:s_end, :])
          test_y.append(df_data[r_begin:r_end, 0].reshape(-1, 1))
          test_index.append(test_sample_index)
          test_sample_index += 1
    else:
      # Electricity is a univariate dataset
      raise NotImplementedError

    self.train_x = np.array(train_x, dtype=np.float16)
    self.train_y = np.array(train_y, dtype=np.float16)
    self.train_index = np.array(train_index)

    self.valid_x = np.array(valid_x, dtype=np.float16)
    self.valid_y = np.array(valid_y, dtype=np.float16)
    self.valid_index = np.array(valid_index)

    self.test_x = np.array(test_x, dtype=np.float16)
    self.test_y = np.array(test_y, dtype=np.float16)
    self.test_index = np.array(test_index)
