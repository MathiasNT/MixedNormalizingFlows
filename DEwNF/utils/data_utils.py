from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import pandas as pd


def get_split_idx_on_day(df, random_state=42):
    # Get unique days from df and create a test / train based on split.
    unique_days = df.obs_day.unique()
    train_days, test_days = train_test_split(unique_days, test_size=0.2, random_state=random_state)
    train_idx = df.index[df.obs_day.isin(train_days)]
    test_idx = df.index[df.obs_day.isin(test_days)]
    return train_idx, test_idx


def searchlog_day_split(df, obs_cols, context_cols, batch_size, cuda_exp):
    # Split data into test and train sets based on the days in the data
    unique_days = df.obs_day.unique()
    train_days, test_days = train_test_split(unique_days, test_size=0.2, random_state=42)
    train_idx = df.index[df.obs_day.isin(train_days)]
    test_idx = df.index[df.obs_day.isin(test_days)]
    df = df.drop('obs_day', axis=1)

    # Normalize data and send to cuda
    # Fit transformation only on the train data
    obs_scaler = StandardScaler().fit(df.loc[train_idx, obs_cols])
    context_scaler = StandardScaler().fit(df.loc[train_idx, context_cols])

    # Transform training data
    scaled_train_obs = obs_scaler.transform(df.loc[train_idx, obs_cols])
    scaled_train_context = context_scaler.transform(df.loc[train_idx, context_cols])
    scaled_train_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_train_context)), dim=1).type(torch.FloatTensor)

    # Transform test data
    scaled_test_obs = obs_scaler.transform(df.loc[test_idx, obs_cols])
    scaled_test_context = context_scaler.transform(df.loc[test_idx, context_cols])
    scaled_test_data = torch.cat((torch.tensor(scaled_test_obs), torch.tensor(scaled_test_context)), dim=1).type(torch.FloatTensor)

    # Send to cuda if necessary
    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, obs_scaler, context_scaler


def searchlog_semisup_day_split(sup_df, unsup_df, obs_cols, semisup_context_cols, sup_context_cols, batch_size, cuda_exp):
    # Split data into test and train sets based on the days in the data
    unique_days = sup_df.obs_day.unique()
    train_days, test_days = train_test_split(unique_days, test_size=0.2, random_state=42)
    train_idx = sup_df.index[sup_df.obs_day.isin(train_days)]
    test_idx = sup_df.index[sup_df.obs_day.isin(test_days)]
    sup_df = sup_df.drop('obs_day', axis=1)

    # Normalize semisup-data and send to cuda
    # Concat the unsupervised obs and context to the train data
    semisup_obs = pd.concat([sup_df.loc[train_idx, obs_cols], unsup_df.loc[:, obs_cols]])
    semisup_context = pd.concat([sup_df.loc[train_idx, semisup_context_cols], unsup_df.loc[:, semisup_context_cols]])

    # Fit transformation of the observations and semisupervised context
    obs_scaler = StandardScaler().fit(semisup_obs)
    semisup_context_scaler = StandardScaler().fit(semisup_context)

    # Fit transformation fo the supervised context
    sup_context_scaler = StandardScaler().fit(sup_df.loc[train_idx, sup_context_cols])

    # Transform training data
    scaled_train_obs = obs_scaler.transform(sup_df.loc[train_idx, obs_cols])
    scaled_train_semisup_context = semisup_context_scaler.transform(sup_df.loc[train_idx, semisup_context_cols])
    scaled_train_sup_context = sup_context_scaler.transform(sup_df.loc[train_idx, sup_context_cols])
    scaled_train_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_train_semisup_context), torch.tensor(scaled_train_sup_context)), dim=1).type(torch.FloatTensor)

    # Transform "extra" data
    scaled_extra_obs = obs_scaler.transform(unsup_df.loc[:,obs_cols])
    scaled_extra_semisup_context = semisup_context_scaler.transform(unsup_df.loc[:, semisup_context_cols])
    scaled_extra_data = torch.cat((torch.tensor(scaled_extra_obs), torch.tensor(scaled_extra_semisup_context)), dim=1).type(torch.FloatTensor)

    # Transform test data
    scaled_test_obs = obs_scaler.transform(sup_df.loc[test_idx, obs_cols])
    scaled_test_semisup_context = semisup_context_scaler.transform(sup_df.loc[test_idx, semisup_context_cols])
    scaled_test_sup_context = sup_context_scaler.transform(sup_df.loc[test_idx, sup_context_cols])
    scaled_test_data = torch.cat((torch.tensor(scaled_test_obs), torch.tensor(scaled_test_semisup_context), torch.tensor(scaled_test_sup_context)), dim=1).type(torch.FloatTensor)

    # Send to cuda if necessary
    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()
        scaled_extra_data = scaled_extra_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    extra_dataloader = DataLoader(scaled_extra_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, extra_dataloader, obs_scaler, semisup_context_scaler, sup_context_scaler


def searchlog_no_weather_day_split(sup_df, unsup_df, obs_cols, semisup_context_cols, batch_size, cuda_exp):
    # Split data into test and train sets based on the days in the data
    unique_days = sup_df.obs_day.unique()
    train_days, test_days = train_test_split(unique_days, test_size=0.2, random_state=42)
    train_idx = sup_df.index[sup_df.obs_day.isin(train_days)]
    test_idx = sup_df.index[sup_df.obs_day.isin(test_days)]
    sup_df = sup_df.drop('obs_day', axis=1)

    # Normalize semisup-data and send to cuda
    # Concat the unsupervised obs and context to the train data
    semisup_obs = pd.concat([sup_df.loc[train_idx, obs_cols], unsup_df.loc[:, obs_cols]])
    semisup_context = pd.concat([sup_df.loc[train_idx, semisup_context_cols], unsup_df.loc[:, semisup_context_cols]])

    # Fit transformation of the observations and semisupervised context
    obs_scaler = StandardScaler().fit(semisup_obs)
    semisup_context_scaler = StandardScaler().fit(semisup_context)


    # Transform training data
    scaled_train_obs = obs_scaler.transform(sup_df.loc[train_idx, obs_cols])
    scaled_train_semisup_context = semisup_context_scaler.transform(sup_df.loc[train_idx, semisup_context_cols])
    scaled_train_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_train_semisup_context)), dim=1).type(torch.FloatTensor)

    # Transform "extra" data
    scaled_extra_obs = obs_scaler.transform(unsup_df.loc[:, obs_cols])
    scaled_extra_semisup_context = semisup_context_scaler.transform(unsup_df.loc[:, semisup_context_cols])
    scaled_extra_data = torch.cat((torch.tensor(scaled_extra_obs), torch.tensor(scaled_extra_semisup_context)), dim=1).type(torch.FloatTensor)

    # Transform test data
    scaled_test_obs = obs_scaler.transform(sup_df.loc[test_idx, obs_cols])
    scaled_test_semisup_context = semisup_context_scaler.transform(sup_df.loc[test_idx, semisup_context_cols])
    scaled_test_data = torch.cat((torch.tensor(scaled_test_obs), torch.tensor(scaled_test_semisup_context)), dim=1).type(torch.FloatTensor)

    # Send to cuda if necessary
    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()
        scaled_extra_data = scaled_extra_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    extra_dataloader = DataLoader(scaled_extra_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, extra_dataloader, obs_scaler, semisup_context_scaler


def searchlog_no_weather_day_split2(sup_df, unsup_df, obs_cols, semisup_context_cols, batch_size, cuda_exp):
    # Split data into test and train sets based on the days in the data
    unique_days = sup_df.obs_day.unique()
    train_days, test_days = train_test_split(unique_days, test_size=0.2, random_state=42)
    train_idx = sup_df.index[sup_df.obs_day.isin(train_days)]
    test_idx = sup_df.index[sup_df.obs_day.isin(test_days)]
    sup_df = sup_df.drop('obs_day', axis=1)

    # Normalize semisup-data and send to cuda
    # Concat the unsupervised obs and context to the train data
    semisup_obs = pd.concat([sup_df.loc[train_idx, obs_cols], unsup_df.loc[:, obs_cols]])
    semisup_context = pd.concat([sup_df.loc[train_idx, semisup_context_cols], unsup_df.loc[:, semisup_context_cols]])

    # Fit transformation of the observations and semisupervised context
    obs_scaler = StandardScaler().fit(semisup_obs)
    semisup_context_scaler = StandardScaler().fit(semisup_context)

    # Transform training data
    scaled_train_obs = obs_scaler.transform(sup_df.loc[train_idx, obs_cols])
    scaled_train_semisup_context = semisup_context_scaler.transform(sup_df.loc[train_idx, semisup_context_cols])
    scaled_train_semisup_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_train_semisup_context)), dim=1).type(torch.FloatTensor)

    # Transform "extra" data
    scaled_extra_obs = obs_scaler.transform(unsup_df.loc[:, obs_cols])
    scaled_extra_semisup_context = semisup_context_scaler.transform(unsup_df.loc[:, semisup_context_cols])
    scaled_extra_data = torch.cat((torch.tensor(scaled_extra_obs), torch.tensor(scaled_extra_semisup_context)), dim=1).type(torch.FloatTensor)

    scaled_train_data = torch.cat((scaled_train_semisup_data, scaled_extra_data))

    # Transform test data
    scaled_test_obs = obs_scaler.transform(sup_df.loc[test_idx, obs_cols])
    scaled_test_semisup_context = semisup_context_scaler.transform(sup_df.loc[test_idx, semisup_context_cols])
    scaled_test_data = torch.cat((torch.tensor(scaled_test_obs), torch.tensor(scaled_test_semisup_context)), dim=1).type(torch.FloatTensor)

    # Send to cuda if necessary
    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, obs_scaler, semisup_context_scaler


def searchlog_unconditional_day_split(sup_df, unsup_df, obs_cols, batch_size, cuda_exp):
    # Split data into test and train sets based on the days in the data
    unique_days = sup_df.obs_day.unique()
    train_days, test_days = train_test_split(unique_days, test_size=0.2, random_state=42)
    train_idx = sup_df.index[sup_df.obs_day.isin(train_days)]
    test_idx = sup_df.index[sup_df.obs_day.isin(test_days)]
    sup_df = sup_df.drop('obs_day', axis=1)

    # Normalize semisup-data and send to cuda
    # Concat the unsupervised obs and context to the train data
    semisup_obs = pd.concat([sup_df.loc[train_idx, obs_cols], unsup_df.loc[:, obs_cols]])

    # Fit transformation of the observations and semisupervised context
    obs_scaler = StandardScaler().fit(semisup_obs)

    # Transform training data
    scaled_train_obs = obs_scaler.transform(sup_df.loc[train_idx, obs_cols])

    # Transform "extra" data
    scaled_extra_obs = obs_scaler.transform(unsup_df.loc[:, obs_cols])

    # concatenate the train and extra data
    scaled_train_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_extra_obs))).float()

    # Transform test data
    scaled_test_data = torch.tensor(obs_scaler.transform(sup_df.loc[test_idx, obs_cols])).float()

    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, obs_scaler


def split_synthetic(df, batch_size, data_size, cuda_exp, random_state=None):
    # Random permute the dataframe to get random splits
    if random_state is None:
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split id
    split_id = int(data_size * 0.8)

    # Take subset of data and split
    df = df[:data_size]
    train_df = df[:split_id]
    test_df = df[split_id:]

    # Fit transform
    obs_scaler = StandardScaler().fit(train_df.iloc[:, 0:2])
    context_scaler = StandardScaler().fit(train_df.iloc[:, 2:])

    # Transform training data
    scaled_train_obs = obs_scaler.transform(train_df.iloc[:, 0:2])
    scaled_train_context = context_scaler.transform(train_df.iloc[:, 2:])
    scaled_train_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_train_context)), dim=1).type(torch.FloatTensor)

    # Transform test data
    scaled_test_obs = obs_scaler.transform(test_df.iloc[:, 0:2])
    scaled_test_context = context_scaler.transform(test_df.iloc[:, 2:])
    scaled_test_data = torch.cat((torch.tensor(scaled_test_obs), torch.tensor(scaled_test_context)), dim=1).type(torch.FloatTensor)

    # Send to cuda if necessary
    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, obs_scaler, context_scaler


def simple_data_split(df, obs_cols, batch_size, cuda_exp):
    # Split data into test and train sets
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

    # Normalize data and send to cuda
    # Fit transformation of the observations
    obs_scaler = StandardScaler().fit(df.loc[train_idx, obs_cols])

    # Transform training data
    scaled_train_data = torch.tensor(obs_scaler.transform(df.loc[train_idx, obs_cols])).float()

    # Transform "extra" data
    scaled_test_data = torch.tensor(obs_scaler.transform(df.loc[test_idx, obs_cols])).float()

    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, obs_scaler


def simple_data_split_conditional(df, obs_cols, context_cols, batch_size, cuda_exp):
    # Split data into test and train sets
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

    # Normalize data and send to cuda
    # Fit transformation of the observations
    obs_scaler = StandardScaler().fit(df.loc[train_idx, obs_cols])
    context_scaler = StandardScaler().fit(df.loc[train_idx, context_cols])

    # Transform training data
    scaled_train_obs = obs_scaler.transform(df.loc[train_idx, obs_cols])
    scaled_train_context = context_scaler.transform(df.loc[train_idx, context_cols])
    scaled_train_data = torch.cat((torch.tensor(scaled_train_obs), torch.tensor(scaled_train_context)), dim=1).type(torch.FloatTensor)

    # Transform "extra" data
    scaled_test_obs = obs_scaler.transform(df.loc[test_idx, obs_cols])
    scaled_test_context = context_scaler.transform(df.loc[test_idx, context_cols])
    scaled_test_data = torch.cat((torch.tensor(scaled_test_obs), torch.tensor(scaled_test_context)), dim=1).type(torch.FloatTensor)

    if cuda_exp:
        scaled_train_data = scaled_train_data.cuda()
        scaled_test_data = scaled_test_data.cuda()

    # Wrap in dataloaders to take care of batching
    train_dataloader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(scaled_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, obs_scaler, context_scaler
