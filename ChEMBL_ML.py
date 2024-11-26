from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import Draw
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


print("Current directory:", os.getcwd())
os.chdir("/Users/anishahinyan/Desktop/HW2")

mydata = pd.read_csv("hwdatachem.csv", sep=';')
print("\nNumber of rows and columns:", mydata.shape)
print("\nColumn names:")
print("\nFirst few rows of key columns:")

mydata = mydata.sample(n=10000, random_state=42)

print("Checking first few SMILES:")
print(mydata['Smiles'].head())

output_file = "sampled_hwdatachem.csv"
mydata.to_csv(output_file, index=False) 
print(f"\nData saved to {output_file}")


print("\nConverting SMILES to molecules...")
mols = []
for s in tqdm(mydata['Smiles']):
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        print(f"Failed to convert SMILES: {s}")
    else:
        mols.append(mol)

print(f"\nSuccessfully converted {len(mols)} molecules")

print("\nCalculating Morgan fingerprints...")
fps_morgan = []
for m in tqdm(mols):
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    fps_morgan.append(fp)


print("\nCalculating Murcko scafolds...")
scaffolds = [MurckoScaffold.GetScaffoldForMol(mol) for mol in tqdm(mols)]
scaffold_smiles = [Chem.MolToSmiles(scaffold) for scaffold in scaffolds]

from collections import Counter
scaffold_counts = Counter(scaffold_smiles)
top_10_scaffolds = scaffold_counts.most_common(10)

print("\nTop 10 scaffolds:")
for scaffold, count in top_10_scaffolds:
    print(f"Count {count}: {scaffold}")

top_scaffold_mols = [Chem.MolFromSmiles(smiles) for smiles, _ in top_10_scaffolds]
img = Draw.MolsToGridImage(top_scaffold_mols, 
                          legends=[f"Count: {count}" for _, count in top_10_scaffolds],
                          molsPerRow=5, 
                          subImgSize=(300,300))
img.save("top_10_scaffolds.png")


print("\nAnalyzig Morgan fingerprint bits...")
all_bits = np.zeros((len(fps_morgan), 2048))
for i, fp in enumerate(fps_morgan):
    for j in range(2048):
        all_bits[i,j] = fp.GetBit(j)

bit_sums = np.sum(all_bits, axis=0)
top_bits = np.argsort(bit_sums)[-10:][::-1]

print("\nTop 10 most common Morgan fingerprint bits:")
for bit, count in zip(top_bits, bit_sums[top_bits]):
    print(f"Bit {bit}: appears in {int(count)} molecules ({(count/len(fps_morgan))*100:.1f}%)")

print("\nFinding example molecules for top bits")
bit_mols = []
legends = []

for bit in top_bits:
    bit = int(bit)
    for mol in tqdm(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        if fp.GetBit(int(bit)):  
            bit_mols.append(mol)
            count = int(bit_sums[bit])
            legends.append(f"Bit {bit}: {count} times")
            break

print("\nDrawing molecules containing top bits")
img = Draw.MolsToGridImage(bit_mols, 
                          legends=legends,
                          molsPerRow=5, 
                          subImgSize=(300,300),
                          returnPNG=False)
img.save("top_10_bits.png")

#Splitting Dataset Into train and test for further training
#Randomly
print("\nerforming random split...")
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    mydata, mydata['Standard Value'], test_size=0.3, random_state=42
)

#Scaffold split
print("\nperforming scaffold split...")
unique_scaffolds = list(set(scaffold_smiles))
n_scaffolds = len(unique_scaffolds)
n_train_scaffolds = int(0.7 * n_scaffolds)

train_scaffolds = unique_scaffolds[:n_train_scaffolds]
test_scaffolds = unique_scaffolds[n_train_scaffolds:]

train_idx = [i for i, scaffold in enumerate(scaffold_smiles) if scaffold in train_scaffolds]
test_idx = [i for i, scaffold in enumerate(scaffold_smiles) if scaffold in test_scaffolds]

data_train_scaffold = mydata.iloc[train_idx]
data_test_scaffold = mydata.iloc[test_idx]

print("\nSplit sizes:")
print(f"Random split - Train: {len(X_train_random)}, Test: {len(X_test_random)}")
print(f"Scaffold split - Train: {len(data_train_scaffold)}, Test: {len(data_test_scaffold)}")

print("splits done")

y = mydata['Standard Value'].apply(lambda x: int(x < 10000)).values

print("\nPerforming random split...")
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    mydata, y, test_size=0.3, random_state=42
)

print("\nPerforming scaffold split...")

unique_scaffolds = list(set(scaffold_smiles))
n_scaffolds = len(unique_scaffolds)
n_train_scaffolds = int(0.7 * n_scaffolds)

train_scaffolds = unique_scaffolds[:n_train_scaffolds]
test_scaffolds = unique_scaffolds[n_train_scaffolds:]

train_idx = [i for i, scaffold in enumerate(scaffold_smiles) if scaffold in train_scaffolds]
test_idx = [i for i, scaffold in enumerate(scaffold_smiles) if scaffold in test_scaffolds]

data_train_scaffold = mydata.iloc[train_idx]
y_train_scaffold = y[train_idx]
data_test_scaffold = mydata.iloc[test_idx]
y_test_scaffold = y[test_idx]

print("\nSplit sizes:")
print(f"Random split - Train: {len(X_train_random)}, Test: {len(X_test_random)}")
print(f"Scaffold split - Train: {len(data_train_scaffold)}, Test: {len(data_test_scaffold)}")

print("\nClass balance (fraction of actives):")
print(f"Random split - Train: {y_train_random.mean():.3f}, Test: {y_test_random.mean():.3f}")
print(f"Scaffold split - Train: {y_train_scaffold.mean():.3f}, Test: {y_test_scaffold.mean():.3f}")

print("Converting fingerprints to arrays...")
X = []
for fp in tqdm(fps_morgan):
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    X.append(arr)
X = np.array(X)

valid_indices = [i for i, mol in enumerate(mols) if mol is not None]


X = X[valid_indices]
y = y[valid_indices]  

print(f"\nShape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

print("\nprforming random split")
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nTraining logistic regression...")
logistic = LogisticRegression(max_iter=10000)

logistic_random = logistic.fit(X_train_random, y_train_random)
print(f"Random split - Train: {logistic_random.score(X_train_random, y_train_random):.3f}, "
      f"Test: {logistic_random.score(X_test_random, y_test_random):.3f}")

logistic_scaffold = logistic.fit(X[train_idx], y_train_scaffold)
print(f"Scaffold split - Train: {logistic_scaffold.score(X[train_idx], y_train_scaffold):.3f}, "
      f"Test: {logistic_scaffold.score(X[test_idx], y_test_scaffold):.3f}")

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)


rf_random = rf.fit(X_train_random, y_train_random)
print(f"Random split - Train: {rf_random.score(X_train_random, y_train_random):.3f}, "
      f"Test: {rf_random.score(X_test_random, y_test_random):.3f}")


rf_scaffold = rf.fit(X[train_idx], y_train_scaffold)
print(f"Scaffold split - Train: {rf_scaffold.score(X[train_idx], y_train_scaffold):.3f}, "
      f"Test: {rf_scaffold.score(X[test_idx], y_test_scaffold):.3f}")


print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)


gb_random = gb.fit(X_train_random, y_train_random)
print(f"Random split - Train: {gb_random.score(X_train_random, y_train_random):.3f}, "
      f"Test: {gb_random.score(X_test_random, y_test_random):.3f}")


gb_scaffold = gb.fit(X[train_idx], y_train_scaffold)
print(f"Scaffold split - Train: {gb_scaffold.score(X[train_idx], y_train_scaffold):.3f}, "
      f"Test: {gb_scaffold.score(X[test_idx], y_test_scaffold):.3f}")

#Cross validation to see if i would get better results using scikit

logistic = LogisticRegression(max_iter=10000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)


print("\nPerforming 5-fold Cross Validation...")


print("\nLogistic Regression CV scores:")
lr_scores = cross_val_score(logistic, X, y, cv=5)
print(f"Mean CV score: {lr_scores.mean():.3f} (+/- {lr_scores.std() * 2:.3f})")


print("\nRandom Forest CV scores:")
rf_scores = cross_val_score(rf, X, y, cv=5)
print(f"Mean CV score: {rf_scores.mean():.3f} (+/- {rf_scores.std() * 2:.3f})")

print("\nGradient Boosting CV scores:")
gb_scores = cross_val_score(gb, X, y, cv=5)
print(f"Mean CV score: {gb_scores.mean():.3f} (+/- {gb_scores.std() * 2:.3f})")
