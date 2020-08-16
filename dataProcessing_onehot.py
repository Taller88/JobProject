import pandas as pd


df=pd.read_csv('total_reduce_indepenVariable_third.csv')

total_names=df.columns.tolist()
categorical_names=['BYS04008', 'BYS04009', 'BYS07002', 'BYS07006', 'BYS07014', 'BYS08001', 'BYS08002', 'BYS08005', 'BYS09002', 'BYS09003', 'BYS22001','BYS22002','BYS23002', 'BYS23003', 'BYS23004', 'BYS24002', 'BYS16001', 'BYS20021', 'BYS27020','BYS08007']
numerical_names=['BYS02003', 'BYS02004', 'BYS02012', 'BYS04007', 'BYS07001', 'BYS07025', 'BYS07026', 'BYS07027', 'BYS07029', 'BYS08003', 'BYS08004', 'BYS09001', 'BYS09004', 'BYS10001', 'BYS10002', 'BYS10003', 'BYS10005', 'BYS11001', 'BYS11002', 'BYS11003', 'BYS11004', 'BYS11005', 'BYS11006', 'BYS13001', 'BYS22003', 'BYS22006', 'BYS22012', 'BYS22013', 'BYS22014', 'BYS22015', 'BYS22017', 'BYS25001', 'BYS25003']
discrete_names=['GENDER', 'BYS03001', 'BYS04010', 'BYS05001', 'BYS05003', 'BYS05006', 'BYS06001', 'BYS06004', 'BYS07003', 'BYS07022', 'BYS12001', 'BYS12004', 'BYS12007', 'BYS12010', 'BYS12013', 'BYS12016', 'BYS12019', 'BYS12022', 'BYS14001', 'BYS14013', 'BYS14025', 'BYS15001', 'BYS20001', 'BYS20007', 'BYS20009', 'BYS20011', 'BYS20013', 'BYS20015', 'BYS20017', 'BYS20018', 'BYS20019', 'BYS20020', 'BYS21001', 'BYS22010', 'BYS24001', 'BYS24005', 'BYS26001', 'BYS26002', 'BYS26003', 'BYS26004', 'BYS26005', 'BYS26006', 'BYS26007', 'BYS26009', 'BYS27001', 'BYS27003', 'BYS27006', 'BYS27012']

except_data=['BYSID','BYHID', 'BYAID', 'BYTID', 'BYSCLASS' ]
#제거이유: 기본 ID와 같은 속성들과 수상종목인데 선택된 값은 0,1로 구성되어 배제


categorical_data=df[categorical_names].astype('str')
numerical_data=df[numerical_names]
numerical_data['BYSID']=df['BYSID']

discrete_data=df[discrete_names]
discrete_data['BYSID']=df['BYSID']
categorical_data_oneHot=pd.get_dummies(categorical_data)
categorical_data_oneHot['BYSID']=df['BYSID']

# concat으로 했을경우 BYSID가 KEY로 되지 않는듯 ...
# result_concat=pd.concat([categorical_data, numerical_data])
result_merge=pd.merge(categorical_data_oneHot, numerical_data, on='BYSID')
result_merge=pd.merge(result_merge,discrete_data, on='BYSID')
print('-------------------------------------')
print(categorical_data_oneHot.shape)
print(numerical_data.shape)
print(discrete_data.shape)
print('-------------------------------------')
print(result_merge)
print('-------------------------------------')
result_merge.drop(['BYSID'], axis='columns', inplace=True)
print(result_merge)
result_merge['J007C']=df['J007C']
result_merge.to_csv("total_onehot_fourth.csv",index=False)
print(result_merge)
print(result_merge['J007C'])
print(result_merge['BYSID'])
