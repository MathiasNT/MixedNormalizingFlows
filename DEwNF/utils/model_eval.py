import torch
import pandas as pd
from tqdm import tqdm


def create_prob_df(shape, bounds, flow_dist, obs_scaler):
    with torch.no_grad():
        nlats, nlons = shape

        lats_array = torch.linspace(start=bounds[1][0], end=bounds[0][0], steps=nlats)
        lons_array = torch.linspace(start=bounds[0][1], end=bounds[1][1], steps=nlons)
        delta_lat = abs((lats_array[1] - lats_array[0]).item())
        delta_lon = abs((lons_array[1] - lons_array[0]).item())
        x, y = torch.meshgrid(lats_array, lons_array)

        points = torch.stack((x.reshape(-1), y.reshape(-1)), axis=1)
        scaled_points = torch.tensor(obs_scaler.transform(points), requires_grad=False).float()

        probs = flow_dist.log_prob(scaled_points.cuda()).cpu().detach().numpy()

        prob_dict = {'latitude': points[:, 0], 'longitude': points[:, 1], 'probs': probs}

        prob_df = pd.DataFrame(prob_dict)
        prob_df = prob_df.sort_values('probs', ascending=False)

    return prob_df, delta_lat, delta_lon


def create_points_df(shape, bounds):
    nlats, nlons = shape
    lats_array = torch.linspace(start=bounds[1][0], end=bounds[0][0], steps=nlats)
    lons_array = torch.linspace(start=bounds[0][1], end=bounds[1][1], steps=nlons)
    delta_lat = abs((lats_array[1] - lats_array[0]).item())
    delta_lon = abs((lons_array[1] - lons_array[0]).item())
    x, y = torch.meshgrid(lats_array, lons_array)
    points = torch.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    point_dict = {'latitude': points[:, 0], 'longitude': points[:, 1]}
    point_df = pd.DataFrame(point_dict)

    return point_df, delta_lat, delta_lon


def calculate_cap_of_model(shape, bounds, dist, df, obs_scaler):
    prob_df, delta_lat, delta_lon = create_prob_df(shape=shape, bounds=bounds, flow_dist=dist, obs_scaler=obs_scaler)
    max_points_arr = []
    max_point_vals_arr = []
    n_covered_arr = []
    cumulative_covered = 0
    percent_covered_arr = []
    n_logs = len(df)

    for index, row in tqdm(prob_df.iterrows()):
        max_point = row.loc[['latitude', 'longitude']].values

        # Find points covered by this hub
        in_lat = (df['user_location_latitude'] >= (max_point[0] - delta_lat / 2)) & (
                    df['user_location_latitude'] < (max_point[0] + delta_lat / 2))
        in_lon = (df['user_location_longitude'] >= (max_point[1] - delta_lon / 2)) & (
                    df['user_location_longitude'] < (max_point[1] + delta_lon / 2))
        in_cover = in_lat & in_lon

        # Calculate how many points is covered
        covered_idxs = in_cover[in_cover].index
        n_covered = len(covered_idxs)
        cumulative_covered += n_covered
        percent_covered = cumulative_covered / n_logs

        # Calculate how many point are still left
        df = df.drop(covered_idxs)

        # Save values
        max_points_arr.append(max_point)
        max_point_vals_arr.append(row['probs'])
        n_covered_arr.append(n_covered)
        percent_covered_arr.append(percent_covered)

    return max_points_arr, max_point_vals_arr, n_covered_arr, percent_covered_arr


def calculate_cap_of_hubs(shape, bounds, logs_df, hubs_df, ranked_hubs_df):
    # Take the hubs dataframe and sort it based on the number of renteals from each hub and get the coordinates also drop all of the NA values
    ranked_hub_coordinates = hubs_df.set_index('id').reindex(ranked_hubs_df.index)[['latitude', 'longitude']].dropna().values

    # Go from hubs locations to nearest grid point, what should i do if multiple hubs have the same grid point?
    points_df, delta_lat, delta_lon = create_points_df(shape=shape, bounds=bounds)

    # Find grid points covered by hubs
    # Also if a hub is close to the same grid point it is not added again.
    hub_grid_points_df = pd.DataFrame(columns=['latitude', 'longitude'])
    for hub in ranked_hub_coordinates:
        in_lat = (points_df['latitude'] >= (hub[0] - delta_lat/2)) & (points_df['latitude'] < (hub[0] + delta_lat/2))
        in_lon = (points_df['longitude'] >= (hub[1] - delta_lon/2)) & (points_df['longitude'] < (hub[1] + delta_lon/2))
        in_cover = in_lat & in_lon
        grid_point = points_df[in_cover].loc[:, ['latitude', 'longitude']]
        hub_grid_points_df = hub_grid_points_df.append(grid_point, ignore_index=True)

    # Do cap calculations
    max_points_arr = []
    max_point_vals_arr = []
    n_covered_arr = []
    cumulative_covered = 0
    percent_covered_arr = []
    n_logs = len(logs_df)

    for index, row in tqdm(hub_grid_points_df.iterrows()):
        max_point = row.loc[['latitude', 'longitude']].values

        # Find points covered by this hub
        in_lat = (logs_df['user_location_latitude'] >= (max_point[0] - delta_lat/2)) & (logs_df['user_location_latitude'] < (max_point[0] + delta_lat/2))
        in_lon = (logs_df['user_location_longitude'] >= (max_point[1] - delta_lon/2)) & (logs_df['user_location_longitude'] < (max_point[1] + delta_lon/2))
        in_cover = in_lat & in_lon

        # Calculate how many points is covered
        covered_idxs = in_cover[in_cover].index
        n_covered = len(covered_idxs)
        cumulative_covered += n_covered
        percent_covered = cumulative_covered/n_logs

        # Calculate how many point are still left
        logs_df = logs_df.drop(covered_idxs)

        # Save values
        max_points_arr.append(max_point)
        n_covered_arr.append(n_covered)
        percent_covered_arr.append(percent_covered)

    return max_points_arr, max_point_vals_arr, n_covered_arr, percent_covered_arr


def calculate_cap_of_random(shape, bounds, dist, df, obs_scaler):
    prob_df, delta_lat, delta_lon = create_prob_df(shape=shape, bounds=bounds, flow_dist = dist, obs_scaler=obs_scaler)
    prob_df = prob_df.sample(frac=1)

    max_points_arr = []
    max_point_vals_arr = []
    n_covered_arr = []
    cumulative_covered = 0
    percent_covered_arr = []
    n_logs = len(df)

    for index, row in tqdm(prob_df.iterrows()):
        max_point = row.loc[['latitude', 'longitude']].values

        # Find points covered by this hub
        in_lat = (df['user_location_latitude'] >= (max_point[0] - delta_lat/2)) & (df['user_location_latitude'] < (max_point[0] + delta_lat/2))
        in_lon = (df['user_location_longitude'] >= (max_point[1] - delta_lon/2)) & (df['user_location_longitude'] < (max_point[1] + delta_lon/2))
        in_cover = in_lat & in_lon
        points = df[in_cover].loc[:, ['user_location_latitude', 'user_location_longitude']].values

        # Calculate how many points is covered
        covered_idxs = in_cover[in_cover].index
        n_covered = len(covered_idxs)
        cumulative_covered += n_covered
        percent_covered = cumulative_covered/n_logs

        # Calculate how many point are still left
        df = df.drop(covered_idxs)

        # Save values
        max_points_arr.append(max_point)
        #covered_dfs.append(covered_df)
        n_covered_arr.append(n_covered)
        percent_covered_arr.append(percent_covered)

    return max_points_arr, max_point_vals_arr, n_covered_arr, percent_covered_arr


def calculate_cap_of_perfect_model(shape, bounds, df):
    points_df, delta_lat, delta_lon = create_points_df(shape, bounds)
    cumulative_covered = 0
    n_covered_arr = []
    percent_covered_arr = []
    n_logs = len(df)
    for index, row in tqdm(points_df.iterrows()):
        point = row.loc[['latitude', 'longitude']].values

        # Find points covered by this hub
        in_lat = (df['user_location_latitude'] >= (point[0] - delta_lat / 2)) & (
                    df['user_location_latitude'] < (point[0] + delta_lat / 2))
        in_lon = (df['user_location_longitude'] >= (point[1] - delta_lon / 2)) & (
                    df['user_location_longitude'] < (point[1] + delta_lon / 2))
        in_cover = in_lat & in_lon

        # Calculate how many points is covered
        covered_idxs = in_cover[in_cover].index
        n_covered = len(covered_idxs)
        n_covered_arr.append(n_covered)
        df = df.drop(covered_idxs)

    points_df['n_covered'] = n_covered_arr
    points_df = points_df.sort_values('n_covered', ascending=False)
    for index, row in tqdm(points_df.iterrows()):
        cumulative_covered += row['n_covered']
        percent_covered = cumulative_covered / n_logs
        percent_covered_arr.append(percent_covered)

    max_points_arr = points_df[['latitude', 'longitude']].values
    n_covered_arr = points_df['n_covered'].values

    return max_points_arr, n_covered_arr, percent_covered_arr