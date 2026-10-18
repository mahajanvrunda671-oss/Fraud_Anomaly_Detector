import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("Current folder:",os.getcwd())
print("Files:",os.listdir())
print("CSV files:",[f for f in os.listdir() if f.lower().endswith("CSV")] )
file_path="creditcard.csv (3)\creditcard.csv"
if os.path.exists(file_path):
    print("Dataset found successfully!")
    df=pd.read_csv(file_path)
    
    print(df.head())
    print("Rows and columns:", df.shape)
    print('\n--- Dataset Info & Schema ---')
    print(df.info())
    
    print("\n---Missing Values Check---")
    print(df.isnull().sum())


    print("\n---Duplicates Rows Check---")
    print(f"Total duplicate rows: {df.duplicated().sum()}")

    print("\n---Summary Statistic---")
    print(df.describe())

    print("\n---Class Distribution---")
    print(df['Class'].value_counts())
    print(df['Class'].value_counts(normalize=True) * 100)

    output_dir="outputs"
    os.makedirs(output_dir, exist_ok=True)

    summary_path=os.path.join(output_dir, 'summary_statistics.csv')
    df.describe().to_csv(summary_path)
    print(f"\n[Saved Artifact] Summary ststistics saved to: {summary_path}")

    plt.figure(figsize=(6,4))
    sns.countplot(x='Class',data=df)
    plt.yscale('log')
    
    plt.title('Distribution of Fraud vs Normal Transactions')
    plt.xlabel('Class (0: Normal, 1:Fraud)')
    plt.ylabel('Count')
    plot_path=os.path.join(output_dir, 'class_distribution_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[Saved Artifact] Plot saved to: {plot_path}")

    print("\n--- Generating Correlation Heatmap ---")
    plt.figure(figsize=(10,8))

    subset_cols=list(df.columns[:10]) + ['Amount','Class']
    corr_matrix=df[subset_cols].corr()

    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap (Subset)')

    heatmap_path=os.path.join(output_dir,'correlation_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')

else:
    print("Dataset not found!")


    

