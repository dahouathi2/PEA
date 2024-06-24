import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os
import pickle


####################################################################################################################
###########################################INTERPOLATION METHODS####################################################

# Function to prepare data for PU
def prepare_data(df):
    df['is_promo'] = df['is_promo'].apply(lambda x: 1 if x is True else (-1 if x is False else np.nan))
    dff = df[['price_range', 'sold_units', '2 for a price', '3 for 2', 'bogof', 'bogshp', 'coupon', 'listing fee', 'online', 'save', 'site fee', 'is_promo']].copy()
    pos_ind = np.where(dff['is_promo'] == 1)[0]
    if len(pos_ind) == 0:
        return None, None, None
    np.random.shuffle(pos_ind)
    pos_sample_len = int(np.ceil(0.1 * len(pos_ind)))
    pos_sample = pos_ind[:pos_sample_len]
    
    dff.reset_index(drop=True, inplace=True)
    dff['class_test'] = -1
    dff.loc[pos_sample, 'class_test'] = 1

    X_data = dff['sold_units'].values.reshape(-1, 1)  # Reshape to 2D array for XGBoost
    y_labeled = dff['class_test'].values
    y_positive = dff['is_promo'].values
    return X_data, y_labeled, y_positive

# Function to fit PU estimator
def fit_PU_estimator(X, y, hold_out_ratio, estimator):
    positives = np.where(y == 1.0)[0]
    hold_out_size = int(np.ceil(len(positives) * hold_out_ratio))
    if hold_out_size == 0:
        return estimator, 1.0  # Handle case where there are no hold-out samples
    np.random.shuffle(positives)
    hold_out = positives[:hold_out_size]
    X_hold_out = X[hold_out]
    X = np.delete(X, hold_out, 0)
    y = np.delete(y, hold_out)
    
    estimator.fit(X, y)
    hold_out_predictions = estimator.predict_proba(X_hold_out)[:, 1]
    c = np.mean(hold_out_predictions)
    return estimator, c

# Function to predict PU probabilities
def predict_PU_prob(X, estimator, prob_s1y1):
    predicted_s = estimator.predict_proba(X)[:, 1]
    return predicted_s / prob_s1y1

# Function to perform positive unlabeling
def positive_unlabeling(df):
    X_data, y_labeled, y_positive = prepare_data(df)
    if X_data is None or y_labeled is None:
        df['predicted_promo'] = df['is_promo']
        return df
    y_labeled[y_labeled == -1] = 0
    predicted = np.zeros(len(X_data))
    learning_iterations = 24

    for index in range(learning_iterations):
        pu_estimator, probs1y1 = fit_PU_estimator(X_data, y_labeled, 0.2, XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        predicted += predict_PU_prob(X_data, pu_estimator, probs1y1)
    
    y_predict = [1 if x > 0.9 else 0 for x in (predicted / learning_iterations)]
    df['predicted_promo'] = y_predict
    return df

# Function to update subtactics and price
def update_subtactics_and_price(df):
    binary_columns = ['2 for a price', '3 for 2', 'bogof', 'bogshp', 'coupon', 'listing fee', 'online', 'save', 'site fee']
    
    # Save the true promo indices where both predicted_promo and is_promo are 1
    true_promo_indices = df[(df['predicted_promo'] == 1) & (df['is_promo'] == 1)].index

    if not true_promo_indices.empty:
        # Compute mean price range for true promo values
        price_range_promo_true = df.loc[true_promo_indices, 'price_range'].mean()

        # Find common values for binary columns using true promo values
        common_values_df = df.loc[true_promo_indices, binary_columns]
        
        if not common_values_df.empty:
            common_values = common_values_df.mode().iloc[0]
        else:
            common_values = pd.Series(0, index=binary_columns)  # Default to 0 if empty
        
        # Ensure no NaNs in common values
        common_values = common_values.fillna(0)

        # Update rows where predicted_promo is 1 and original is_promo was NaN
        promo_indices = df[(df['predicted_promo'] == 1) & (df['is_promo'].isna())].index
        df.loc[promo_indices, 'price_range'] = price_range_promo_true
        for col in binary_columns:
            df.loc[promo_indices, col] = common_values[col]
    
    # Set subtactics and price to zero where predicted_promo is 0
    non_promo_indices = df[df['predicted_promo'] == 0].index
    df.loc[non_promo_indices, binary_columns] = 0
    df.loc[non_promo_indices, 'price_range'] = 0

    return df

# Apply the process to each ean_global_channel group
def process_group(group):
    group = positive_unlabeling(group)
    group = update_subtactics_and_price(group)
    return group


def import_true_promo(client, zero_percent, month, num_weeks,channel=None, fill_discontinuity=False, keep_non_promo=False):

    """
    This function download data From gcp
    Options:
    - zero_percentage os sales values == 0
    - month to do the split train/test
    - num_weeks minimal we'll add assert num_weeks>3*prediction_length
    - channel Both, offline, online
    -fill_discontinuity: add the product that has discounuity in values and interpolate them
    - keep non_propo: true means we also keep the product that has no promotions during the whole period
    """
    def query(zero_percent, keep_non_promo = False):
        

        if keep_non_promo:
            a = """
                WITH MinPromoDate AS (
                    SELECT
                        MIN(end_date) AS min_date
                    FROM
                        `itg-bpma-gbl-ww-np.bpma_ds_c2_exposed_eu_np.pnl_details_sellout_no_fakes`
                    WHERE
                        is_promo = TRUE
                ),
                TransformedData AS (
                    SELECT
                        start_date,
                        end_date,
                        sub_axis,
                        ean,
                        global_channel_type,
                        seasonality_index,
                        CASE
                            WHEN is_promo = FALSE AND end_date >= (SELECT min_date FROM MinPromoDate) THEN 0
                            ELSE price_range
                        END AS price_range,
                        sold_units,
                        CASE
                            WHEN is_promo = FALSE AND end_date >= (SELECT min_date FROM MinPromoDate) THEN ''
                            ELSE sub_tactic
                        END AS sub_tactic,
                        CASE
                            WHEN is_promo = FALSE AND end_date < (SELECT min_date FROM MinPromoDate) THEN NULL
                            ELSE is_promo
                        END AS is_promo
                    FROM
                        `itg-bpma-gbl-ww-np.bpma_ds_c2_exposed_eu_np.pnl_details_sellout_no_fakes`
                    WHERE
                        ean IS NOT NULL AND
                        end_date IS NOT NULL
                ),
                EANThreshold AS (
                    SELECT
                        ean,
                        global_channel_type,
                        SUM(CASE WHEN sold_units = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS ZeroPercent
                    FROM
                        TransformedData
                    GROUP BY
                        ean,
                        global_channel_type
                    HAVING
                        ZeroPercent <= {}
                )
                SELECT
                    td.start_date,
                    td.end_date,
                    td.sub_axis,
                    td.ean,
                    td.global_channel_type,
                    td.seasonality_index,
                    td.price_range,
                    td.is_promo,
                    td.sub_tactic,
                    td.sold_units
                FROM
                    TransformedData td
                JOIN
                    EANThreshold et
                ON
                    td.ean = et.ean
                    AND td.global_channel_type = et.global_channel_type
                WHERE
                    td.end_date >= (SELECT min_date FROM MinPromoDate)
                """.format(zero_percent)
        else:
            a = """
                WITH MinPromoDate AS (
                    SELECT
                        MIN(end_date) AS min_date
                    FROM
                        `itg-bpma-gbl-ww-np.bpma_ds_c2_exposed_eu_np.pnl_details_sellout_no_fakes`
                    WHERE
                        is_promo = TRUE
                ),
                TransformedData AS (
                    SELECT
                        start_date,
                        end_date,
                        sub_axis,
                        ean,
                        global_channel_type,
                        seasonality_index,
                        CASE
                            WHEN is_promo = FALSE AND end_date >= (SELECT min_date FROM MinPromoDate) THEN 0
                            ELSE price_range
                        END AS price_range,
                        sold_units,
                        CASE
                            WHEN is_promo = FALSE AND end_date >= (SELECT min_date FROM MinPromoDate) THEN ''
                            ELSE sub_tactic
                        END AS sub_tactic,
                        CASE
                            WHEN is_promo = FALSE AND end_date < (SELECT min_date FROM MinPromoDate) THEN NULL
                            ELSE is_promo
                        END AS is_promo
                    FROM
                        `itg-bpma-gbl-ww-np.bpma_ds_c2_exposed_eu_np.pnl_details_sellout_no_fakes`
                    WHERE
                        ean IS NOT NULL AND
                        end_date IS NOT NULL
                ),
                EANThreshold AS (
                    SELECT
                        ean,
                        global_channel_type,
                        SUM(CASE WHEN sold_units = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS ZeroPercent
                    FROM
                        TransformedData
                    GROUP BY
                        ean,
                        global_channel_type
                    HAVING
                        ZeroPercent <= {}
                ),
                PromoFilter AS (
                    SELECT
                        ean,
                        global_channel_type,
                        SUM(CASE WHEN is_promo = TRUE THEN 1 ELSE 0 END) > 0 AS has_promo
                    FROM
                        TransformedData
                    GROUP BY
                        ean,
                        global_channel_type
                    HAVING
                        has_promo
                )
                SELECT
                    td.start_date,
                    td.end_date,
                    td.sub_axis,
                    td.ean,
                    td.global_channel_type,
                    td.seasonality_index,
                    td.price_range,
                    td.is_promo,
                    td.sub_tactic,
                    td.sold_units
                FROM
                    TransformedData td
                JOIN
                    EANThreshold et
                ON
                    td.ean = et.ean
                    AND td.global_channel_type = et.global_channel_type
                JOIN
                    PromoFilter pf
                ON
                    td.ean = pf.ean
                    AND td.global_channel_type = pf.global_channel_type
                WHERE
                    td.end_date >= (SELECT min_date FROM MinPromoDate)
                """.format(zero_percent)
        if channel=='Online': 
            a+="""AND td.global_channel_type = 'Online'
            ORDER BY
                td.end_date;"""
        elif channel=='Offline':
            a+="""AND td.global_channel_type = 'Offline'
            ORDER BY
                td.end_date;"""
        else:
            a+="""
            ORDER BY
                td.end_date;"""
        return a

    data['ean_global_channel'] = data['ean'] + '_' + data['global_channel_type']
    print("number of products before preprocessing", data["ean_global_channel"].unique().shape[0])


    # Step 1: Count unique end dates for each ean_global_channel
    unique_dates = data.groupby('ean_global_channel')['end_date'].nunique().reset_index()

    # Step 2: Filter to find ean_global_channels with more than or equal to num_weeks unique dates
    valid_ean_global_channels = unique_dates[unique_dates['end_date'] >= num_weeks]['ean_global_channel']

    # Step 3: Filter the original DataFrame to include only these ean_global_channels
    data = data[data['ean_global_channel'].isin(valid_ean_global_channels)]

    data['sub_tactic'] = data['sub_tactic'].str.lower().str.strip()

    def aggregate_subtactics(series):
        if series is None or all(pd.isnull(series)): 
            return ''
        all_subtactics = set()
        for items in series.dropna():
            tactics = set(item.strip() for item in items.split(','))
            all_subtactics.update(tactics)
        return ', '.join(sorted(all_subtactics))

    def custom_price_range(series):
        return series.mean(skipna=True) if not series.isnull().all() else np.nan

    aggregated_data = data.groupby(['start_date', 'end_date', 'ean_global_channel']).agg({
        'is_promo': 'first',
        'price_range': custom_price_range,
        'sub_tactic': aggregate_subtactics,
        'sub_axis': 'first',
        'seasonality_index': 'first',
        'sold_units': 'first'
    }).reset_index()

    aggregated_data.drop_duplicates(inplace=True)
    print("How many ean_global_channel_type:", aggregated_data.ean_global_channel.unique().shape[0])
    if aggregated_data.ean_global_channel.unique().shape[0] == 0:
        raise ValueError("Error: No unique ean_global_channel values found.")
    one_hot_encoded_data = aggregated_data['sub_tactic'].str.get_dummies(', ')
    empty_sub_tactic_indices = aggregated_data[aggregated_data['sub_tactic'] == ''].index
    one_hot_encoded_data.loc[empty_sub_tactic_indices] = 0

    final_data = pd.concat([aggregated_data, one_hot_encoded_data], axis=1)
    final_data.drop(['sub_tactic'], axis=1, inplace=True)

    def shuffle_and_sort(group):
        shuffled_group = group.sample(frac=1).reset_index(drop=True)
        sorted_group = shuffled_group.sort_values('end_date')
        return sorted_group

    final_data = final_data.groupby(['ean_global_channel', 'sub_axis'], group_keys=False).apply(shuffle_and_sort).reset_index(drop=True)
    final_data.drop(["start_date"], axis=1, inplace=True)
    final_data['seasonality_index'] = final_data['seasonality_index'].fillna(method='bfill')

    if fill_discontinuity:
        #  We Create a full date range for each ean_global_channel,
        full_data = []
        for name, group in final_data.groupby(['ean_global_channel']):
            group['end_date'] = pd.to_datetime(group['end_date'])
            group.set_index('end_date', inplace=True)
            full_range = pd.date_range(start= group.index.min(), end=group.index.max(), freq='W-SAT') #'10-08-2022'
            group = group.reindex(full_range).ffill().reset_index().rename(columns={'index': 'end_date'})
            full_data.append(group)
        final_data = pd.concat(full_data).reset_index(drop=True)

    result = final_data.groupby('ean_global_channel')['end_date'].agg(['min', 'max']).reset_index().sort_values(by='max', ascending=False)
    max_date_first_row = result.iloc[0]["max"]
    filtered_channels = result[result['max'] < max_date_first_row]['ean_global_channel'].reset_index(drop=True)

    final_data = final_data[~final_data['ean_global_channel'].isin(filtered_channels)]
    final_data["end_date"] = pd.to_datetime(final_data["end_date"])
    final_data["year"] = final_data["end_date"].dt.year
    final_data["month"] = final_data["end_date"].dt.month
    final_data["week"] = final_data["end_date"].dt.isocalendar().week

    train_set = final_data.loc[((final_data['year'] == 2022) | ((final_data['year'] == 2023) & (final_data['month'] <= month)))]
    test_set = final_data.loc[((final_data['year'] == 2023) & (final_data['month'] > month)) | (final_data['year'] == 2024)]


    ean_test_date = test_set.groupby("ean_global_channel").end_date.count().reset_index().sort_values('end_date')
    max_date_first_row = ean_test_date.iloc[-1]["end_date"]

    # Filter the ean_global_channel in result where max date is less than the max date of the first row
    filtered_channels = ean_test_date[ean_test_date['end_date'] < max_date_first_row]['ean_global_channel'].reset_index(drop=True)

    # Filter the original DataFrame based on the filtered ean_global_channel
    final_data = final_data[~final_data['ean_global_channel'].isin(filtered_channels)]

    train_set = final_data.loc[((final_data['year'] == 2022) | ((final_data['year'] == 2023) & (final_data['month'] <= month)))]
    test_set = final_data.loc[((final_data['year'] == 2023) & (final_data['month'] > month)) | (final_data['year'] == 2024)]
    print("final data product (if changed we remove discontinuity)", final_data.ean_global_channel.unique().shape[0] )
    ean_test_date = test_set.groupby("ean_global_channel").end_date.count().reset_index().sort_values('end_date')
    max_date_first_row = ean_test_date.iloc[-1]["end_date"]
    min_date_first_row = ean_test_date.iloc[0]["end_date"]
    print("prediction length:", max_date_first_row)
    assert min_date_first_row == max_date_first_row , "min_date_first_row != max_date_first_row"


    return final_data, train_set, test_set, max_date_first_row


def import_all(client, zero_percent, month,num_weeks, channel=None, fill_discontinuity=False, keep_non_promo=False, interpolation_method=True):
    def query(zero_percent, keep_non_promo = False):
        if keep_non_promo:
            a = """
                WITH MinPromoDate AS (
                    SELECT
                        MIN(end_date) AS min_date
                    FROM
                        `itg-bpma-gbl-ww-np.bpma_ds_c2_exposed_eu_np.pnl_details_sellout_no_fakes`
                    WHERE
                        is_promo = TRUE
                ),
                TransformedData AS (
                    SELECT
                        start_date,
                        end_date,
                        sub_axis,
                        ean,
                        global_channel_type,
                        seasonality_index,
                        CASE
                            WHEN is_promo = FALSE AND end_date >= (SELECT min_date FROM MinPromoDate) THEN 0
                            ELSE price_range
                        END AS price_range,
                        sold_units,
                        CASE
                            WHEN is_promo = FALSE AND end_date >= (SELECT min_date FROM MinPromoDate) THEN ''
                            ELSE sub_tactic
                        END AS sub_tactic,
                        CASE
                            WHEN is_promo = FALSE AND end_date < (SELECT min_date FROM MinPromoDate) THEN NULL
                            ELSE is_promo
                        END AS is_promo
                    FROM
                        `itg-bpma-gbl-ww-np.bpma_ds_c2_exposed_eu_np.pnl_details_sellout_no_fakes`
                    WHERE
                        ean IS NOT NULL AND
                        end_date IS NOT NULL
                ),
                EANThreshold AS (
                    SELECT
                        ean,
                        global_channel_type,
                        SUM(CASE WHEN sold_units = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS ZeroPercent
                    FROM
                        TransformedData
                    GROUP BY
                        ean,
                        global_channel_type
                    HAVING
                        ZeroPercent <= {}
                )
                SELECT
                    td.start_date,
                    td.end_date,
                    td.sub_axis,
                    td.ean,
                    td.global_channel_type,
                    td.seasonality_index,
                    td.price_range,
                    td.is_promo,
                    td.sub_tactic,
                    td.sold_units
                FROM
                    TransformedData td
                JOIN
                    EANThreshold et
                ON
                    td.ean = et.ean
                    AND td.global_channel_type = et.global_channel_type
                """.format(zero_percent)
        else:
            a = """
            WITH MinPromoDate AS (
                SELECT
                    MIN(end_date) AS min_date
                FROM
                    `itg-bpma-gbl-ww-np.bpma_ds_c2_exposed_eu_np.pnl_details_sellout_no_fakes`
                WHERE
                    is_promo = TRUE
            ),
            TransformedData AS (
                SELECT
                    start_date,
                    end_date,
                    sub_axis,
                    ean,
                    global_channel_type,
                    seasonality_index,
                    CASE
                        WHEN is_promo = FALSE AND end_date >= (SELECT min_date FROM MinPromoDate) THEN 0
                        ELSE price_range
                    END AS price_range,
                    sold_units,
                    CASE
                        WHEN is_promo = FALSE AND end_date >= (SELECT min_date FROM MinPromoDate) THEN ''
                        ELSE sub_tactic
                    END AS sub_tactic,
                    CASE
                        WHEN is_promo = FALSE AND end_date < (SELECT min_date FROM MinPromoDate) THEN NULL
                        ELSE is_promo
                    END AS is_promo
                FROM
                    `itg-bpma-gbl-ww-np.bpma_ds_c2_exposed_eu_np.pnl_details_sellout_no_fakes`
                WHERE
                    ean IS NOT NULL AND
                    end_date IS NOT NULL
            ),
            EANThreshold AS (
                SELECT
                    ean,
                    global_channel_type,
                    SUM(CASE WHEN sold_units = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS ZeroPercent
                FROM
                    TransformedData
                GROUP BY
                    ean,
                    global_channel_type
                HAVING
                    ZeroPercent <= {}
            ),
            PromoEANs AS (
                SELECT
                    ean,
                    global_channel_type,
                    SUM(CASE WHEN is_promo = TRUE THEN 1 ELSE 0 END) > 0 AS has_promo
                FROM
                    TransformedData
                GROUP BY
                    ean,
                    global_channel_type
                HAVING
                    has_promo
            )
            SELECT
                td.start_date,
                td.end_date,
                td.sub_axis,
                td.ean,
                td.global_channel_type,
                td.seasonality_index,
                td.price_range,
                td.is_promo,
                td.sub_tactic,
                td.sold_units
            FROM
                TransformedData td
            JOIN
                EANThreshold et
            ON
                td.ean = et.ean
                AND td.global_channel_type = et.global_channel_type
            JOIN
                PromoEANs pe
            ON
                td.ean = pe.ean
                AND td.global_channel_type = pe.global_channel_type
            """.format(zero_percent)

        if channel=='Online': 
            a+="""where td.global_channel_type = 'Online'
            ORDER BY
                td.end_date;"""
        elif channel=='Offline':
            a+="""where td.global_channel_type = 'Offline'
            ORDER BY
                td.end_date;"""
        else:
            a+="""
            ORDER BY
                td.end_date;"""
        return a

    data =client.query_and_wait(query(zero_percent, keep_non_promo)).to_dataframe()
    data['ean_global_channel'] = data['ean'] + '_' + data['global_channel_type']
    print("number of products before preprocessing", data["ean_global_channel"].unique().shape[0])



    

    # Step 1: Count unique end dates for each ean_global_channel
    unique_dates = data.groupby('ean_global_channel')['end_date'].nunique().reset_index()

    # Step 2: Filter to find ean_global_channels with more than or equal to num_weeks unique dates
    valid_ean_global_channels = unique_dates[unique_dates['end_date'] >= num_weeks]['ean_global_channel']

    # Step 3: Filter the original DataFrame to include only these ean_global_channels
    data = data[data['ean_global_channel'].isin(valid_ean_global_channels)]

    # Convert 'sold_units' to float
    data["sold_units"] = data["sold_units"].astype(float)

    # Sort the data
    data = data.sort_values(by=["end_date", "global_channel_type", "ean"])
    data['sub_tactic'] = data['sub_tactic'].str.lower().str.strip()

    def aggregate_subtactics(series):
        if series is None or all(pd.isnull(series)): 
            return ''
        all_subtactics = set()
        for items in series.dropna():
            tactics = set(item.strip() for item in items.split(','))
            all_subtactics.update(tactics)
        return ', '.join(sorted(all_subtactics))

    def custom_price_range(series):
        return series.mean(skipna=True) if not series.isnull().all() else np.nan

    aggregated_data = data.groupby(['start_date', 'end_date', 'ean_global_channel']).agg({
        'is_promo': 'first',
        'price_range': custom_price_range,
        'sub_tactic': aggregate_subtactics,
        'sub_axis': 'first',
        'seasonality_index': 'first',
        'sold_units': 'first'
    }).reset_index()

    aggregated_data.drop_duplicates(inplace=True)
    print("How many ean_global_channel_type:", aggregated_data.ean_global_channel.unique().shape[0])
    if aggregated_data.ean_global_channel.unique().shape[0] == 0:
        raise ValueError("Error: No unique ean_global_channel values found.")
    one_hot_encoded_data = aggregated_data['sub_tactic'].str.get_dummies(', ')
    empty_sub_tactic_indices = aggregated_data[aggregated_data['sub_tactic'] == ''].index
    one_hot_encoded_data.loc[empty_sub_tactic_indices] = 0

    final_data = pd.concat([aggregated_data, one_hot_encoded_data], axis=1)
    final_data.drop(['sub_tactic'], axis=1, inplace=True)

    def shuffle_and_sort(group):
        shuffled_group = group.sample(frac=1).reset_index(drop=True)
        sorted_group = shuffled_group.sort_values('end_date')
        return sorted_group

    final_data = final_data.groupby(['ean_global_channel', 'sub_axis'], group_keys=False).apply(shuffle_and_sort).reset_index(drop=True)
    final_data.drop(["start_date"], axis=1, inplace=True)
    final_data['seasonality_index'] = final_data['seasonality_index'].fillna(method='bfill')

    if fill_discontinuity:
        #  We Create a full date range for each ean_global_channel,
        full_data = []
        for name, group in final_data.groupby(['ean_global_channel']):
            group['end_date'] = pd.to_datetime(group['end_date'])
            group.set_index('end_date', inplace=True)
            full_range = pd.date_range(start= group.index.min(), end=group.index.max(), freq='W-SAT') #'10-08-2022'
            group = group.reindex(full_range).ffill().reset_index().rename(columns={'index': 'end_date'})
            full_data.append(group)
        final_data = pd.concat(full_data).reset_index(drop=True)

    result = final_data.groupby('ean_global_channel')['end_date'].agg(['min', 'max']).reset_index().sort_values(by='max', ascending=False)
    max_date_first_row = result.iloc[0]["max"]
    filtered_channels = result[result['max'] < max_date_first_row]['ean_global_channel'].reset_index(drop=True)

    final_data = final_data[~final_data['ean_global_channel'].isin(filtered_channels)]
    final_data["end_date"] = pd.to_datetime(final_data["end_date"])
    final_data["year"] = final_data["end_date"].dt.year
    final_data["month"] = final_data["end_date"].dt.month
    final_data["week"] = final_data["end_date"].dt.isocalendar().week

    train_set = final_data.loc[((final_data['year'] <= 2023) | ((final_data['year'] == 2023) & (final_data['month'] <= month)))]
    test_set = final_data.loc[((final_data['year'] == 2023) & (final_data['month'] > month)) | (final_data['year'] == 2024)]


    ean_test_date = test_set.groupby("ean_global_channel").end_date.count().reset_index().sort_values('end_date')
    max_date_first_row = ean_test_date.iloc[-1]["end_date"]

    # Filter the ean_global_channel in result where max date is less than the max date of the first row
    filtered_channels = ean_test_date[ean_test_date['end_date'] < max_date_first_row]['ean_global_channel'].reset_index(drop=True)

    # Filter the original DataFrame based on the filtered ean_global_channel
    final_data = final_data[~final_data['ean_global_channel'].isin(filtered_channels)]

    train_set = final_data.loc[((final_data['year'] <= 2022) | ((final_data['year'] == 2023) & (final_data['month'] <= month)))]
    test_set = final_data.loc[((final_data['year'] == 2023) & (final_data['month'] > month)) | (final_data['year'] == 2024)]
    print("final data product (if changed we remove discontinuity)", final_data.ean_global_channel.unique().shape[0] )
    ean_test_date = test_set.groupby("ean_global_channel").end_date.count().reset_index().sort_values('end_date')
    max_date_first_row = ean_test_date.iloc[-1]["end_date"]
    min_date_first_row = ean_test_date.iloc[0]["end_date"]
    print("prediction length:", max_date_first_row)
    assert min_date_first_row == max_date_first_row , "min_date_first_row != max_date_first_row"

    ##################################################################################################
    #######################INTERPOLATION STEP#########################################################
    print("Interpolation step starting now")
    if interpolation_method==False:
        data=final_data.copy()
        data['is_promo'] = data['is_promo'].apply(lambda x: 1 if x is True else (0 if x is False else np.nan))
        # Encoding categorical variables
        data['sub_axis_encoded'] = LabelEncoder().fit_transform(data['sub_axis'])
        data['sold_units'] = pd.to_numeric(data['sold_units'], errors='coerce')

        # Separate the dataset into training and prediction sets
        train_df = data[data['is_promo'].notna()]
        predict_df = data[data['is_promo'].isna()]

        # Split the training data into features and labels
        X = train_df[['sub_axis_encoded', 'sold_units']]
        y = train_df['is_promo']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='auc', colsample_bytree=1.0, eta=0.1, max_depth=6, min_child_weight=5, subsample=1.0)
        xgb_model.fit(X_train, y_train)

        # Predict on the testing set
        y_pred = xgb_model.predict(X_test)

        # Print the classification report and ROC-AUC score
        print(classification_report(y_test, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

        xgb_model.fit(X, y)
        # Predict on the unlabeled data
        X_predict = predict_df[['sub_axis_encoded', 'sold_units']]
        predict_df['is_promo'] = xgb_model.predict(X_predict)

        # Merge the predictions back into the original dataset
        data.update(predict_df)
        def update_subtactics_and_price_(df):
            binary_columns = ['2 for a price', '3 for 2', 'bogof', 'bogshp', 'coupon', 'listing fee', 'online', 'save', 'site fee']
            
            # Save the original promo indices and price range for later use
            original_promo_indices = df[(df['is_promo'] == 1) & (~df['price_range'].isna())].index

            if not original_promo_indices.empty:
                price_range_promo_true = df.loc[original_promo_indices, 'price_range'].mean()

                # Find common values for binary columns using the original promo values
                common_values_df = df.loc[original_promo_indices, binary_columns]

                
                if not common_values_df.empty:
                    common_values = common_values_df.mode().iloc[0]
                else:
                    common_values = pd.Series(0, index=binary_columns)  # Default to 0 if empty
                
                common_values = common_values.fillna(0)  # Ensure no NaNs in common values

            
                # Update rows where is_promo is 1 and original is_promo was NaN
                promo_indices = df[(df['is_promo'] == 1) & (df['price_range'].isna())].index

                
                df.loc[promo_indices, 'price_range'] = price_range_promo_true
                for col in binary_columns:
                    df.loc[promo_indices, col] = common_values[col]
            
            # Set subtactics and price to zero where is_promo is 0
            non_promo_indices = df[df['is_promo'] == 0].index

            
            df.loc[non_promo_indices, binary_columns] = 0
            df.loc[non_promo_indices, 'price_range'] = 0

            if original_promo_indices.empty:
                print(df.ean_global_channel.iloc[0])
            return df
        # Apply the function to update subtactics and price_range based on the new predictions
        result = data.groupby('ean_global_channel').apply(update_subtactics_and_price_).reset_index(drop=True)
        result = result.drop(["sub_axis_encoded"], axis=1)
    else :
        data = final_data.copy()
        result = data.groupby('ean_global_channel').apply(process_group).reset_index(drop=True)
        result['is_promo'] = result['predicted_promo']
        result = result.drop(["predicted_promo"], axis=1)
    
    print("Interpolation step is done")
    ##################################################################################################
    #######################SPLITTING##################################################################
    final_data = result.copy()
    final_data = final_data[~final_data['ean_global_channel'].isin(filtered_channels)]

    train_set = final_data.loc[((final_data['year'] <= 2022) | ((final_data['year'] == 2023) & (final_data['month'] <= month)))]
    test_set = final_data.loc[((final_data['year'] == 2023) & (final_data['month'] > month)) | (final_data['year'] == 2024)]    

    assert max_date_first_row* 3 <num_weeks, "num weeks should be higher than 3 times prediction length"
    return final_data, train_set, test_set, max_date_first_row


def generate_standardization_dicts(data, id_col='ean_global_channel', target_col='sales'):
    """
    Generate dictionaries with means and standard deviations for each id in the data.
    """
    data = data.rename(columns={id_col: 'id'})
    mean_dict = {}
    std_dict = {}

    for id_value, group in data.groupby('id'):
        means = group.mean()
        stds = group.std()
        # Replace zero standard deviations with one
        stds = stds.replace(0, 1)
        mean_dict[id_value] = means.to_dict()
        std_dict[id_value] = stds.to_dict()
    
    ids = list(mean_dict.keys())
    
    return mean_dict, std_dict, ids


def check_saved_standardization_data(path):
    """
    Check if the saved mean_dict, std_dict, and ids files exist in the given path.
    Returns True if all files exist, False otherwise.
    """
    mean_dict_path = os.path.join(path, 'mean_dict.pkl')
    std_dict_path = os.path.join(path, 'std_dict.pkl')
    ids_path = os.path.join(path, 'ids.pkl')

    return os.path.exists(mean_dict_path) and os.path.exists(std_dict_path) and os.path.exists(ids_path)

def save_standardization_data(mean_dict, std_dict, ids, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/mean_dict.pkl', 'wb') as f:
        pickle.dump(mean_dict, f)
    with open(f'{path}/std_dict.pkl', 'wb') as f:
        pickle.dump(std_dict, f)
    with open(f'{path}/ids.pkl', 'wb') as f:
        pickle.dump(ids, f)

def load_standardization_data(path):
    with open(f'{path}/mean_dict.pkl', 'rb') as f:
        mean_dict = pickle.load(f)
    with open(f'{path}/std_dict.pkl', 'rb') as f:
        std_dict = pickle.load(f)
    with open(f'{path}/ids.pkl', 'rb') as f:
        ids = pickle.load(f)
    return mean_dict, std_dict, ids


def delete_saved_standardization_data(path):
    """
    Delete the saved mean_dict, std_dict, and ids files if they exist in the given path.
    """
    mean_dict_path = os.path.join(path, 'mean_dict.pkl')
    std_dict_path = os.path.join(path, 'std_dict.pkl')
    ids_path = os.path.join(path, 'ids.pkl')

    if os.path.exists(mean_dict_path):
        os.remove(mean_dict_path)
        print(f"Deleted {mean_dict_path}")
    if os.path.exists(std_dict_path):
        os.remove(std_dict_path)
        print(f"Deleted {std_dict_path}")
    if os.path.exists(ids_path):
        os.remove(ids_path)
        print(f"Deleted {ids_path}")












