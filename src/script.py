# if __name__ == "__main__":
#     x_data = load_data(f"{os.getcwd()}/data/train_feat.csv")
#     y_data = load_data(f"{os.getcwd()}/data/train_output.csv")
#     y_train = load_data(f"{os.getcwd()}/data/y_train.csv")
#     y_valid = load_data(f"{os.getcwd()}/data/y_valid.csv")
#     public_train = load_data(f"{os.getcwd()}/data/test_feat.csv")
#     print(f"Public train dataset shape: {public_train.shape}")
#     sub_data = load_data(f"{os.getcwd()}/data/public_private_submission_template.csv")
#     private_train = load_data(
#         f"{os.getcwd()}/data/private_dataset.csv"
#     ).drop(columns=['ID'])[public_train.columns]
#     print(f"Private train dataset shape: {private_train.shape}")
#     x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, ratio=0.8)
#     print(
#         f"Train shape: {x_train.shape}, {y_train.shape}, \
#             Valid shape: {x_valid.shape}, {y_valid.shape}"
#     )
#     y_train, y_valid = logarithm(y_train), logarithm(y_valid)
#     stack_model = stacking()
#     stack_model.fit(x_train, y_train)
#     y_pred = stack_model.predict(x_valid)
#     y_pred, y_valid = np.exp(y_pred), np.exp(y_valid)
#     mape = mean_absolute_percentage_error(y_valid, y_pred)
#     print(f"MAPE: {mape * 100}")
#     public_pred = np.exp(stack_model.predict(public_train))
#     private_pred = np.exp(stack_model.predict(private_train))
#     sub_data['predicted_price'][:5876] = public_pred
#     sub_data['predicted_price'][5876:] = private_pred
#     sub_data.to_csv(f"{os.getcwd()}/data/public_private_submission_stack_v1.csv", index=False)



# for external_datas in tqdm(
    #     os.listdir(
    #         f"{os.getcwd()}/data/external_data"
    #     )
    # ):
    #     file_name = external_datas.split('.')[0].replace('資料', '')
    #     md = mean_dist.MeanDist(
    #         f"{os.getcwd()}/data/external_data/{external_datas}",
    #         TARGET_PATH,
    #         args.k,
    #         file_name
    #     )
    #     md.update_dataframe(
    #         column_name=f"mean_distance_to{file_name}"
    #     ).to_csv(
    #         f"{os.getcwd()}/data/small_training_data.csv",
    #         index=False
    #     )

    # new_social_economic_feature = soc_econ.add_social_economic_feature(
    #     *args
    # )

    # nfac = n_facilities_v2.NFacilities(
    #     f"{os.getcwd()}/data/external_data/ATM資料.csv",
    #     f"{os.getcwd()}/data/small_training_data.csv",
    #     args.radius
    # )
    # nfac.main().to_csv(
    #     f"{os.getcwd()}/data/small_training_data.csv",
    #     index=False
    # )