import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve)
sns.set_theme(style="whitegrid"); RS = 42

# ---------- 1. Data preparation ----------
data = load_breast_cancer(as_frame=True)
X = data.data.copy(); X.columns = [c.replace(" ","_") for c in X.columns]
y = (data.target == 0).astype(int)           # 1 = malignant (the class we must not miss), 0 = benign
print("Shape:", X.shape, "| missing:", int(X.isnull().sum().sum()), "| duplicates:", int(X.duplicated().sum()))
print("Malignant share: %.1f%%" % (y.mean()*100))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
print("Train:", X_tr.shape, "Test:", X_te.shape, "| malignant in train/test: %.1f%% / %.1f%%" % (y_tr.mean()*100, y_te.mean()*100))

# ---------- 2. Models (scaler lives inside the pipeline so it is fitted on training folds only) ----------
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
lr = GridSearchCV(Pipeline([("sc",StandardScaler()),("m",LogisticRegression(max_iter=5000))]),
                  {"m__C":[0.001,0.01,0.1,1,10,100]}, cv=cv, scoring="recall").fit(X_tr,y_tr)
dt = GridSearchCV(Pipeline([("sc",StandardScaler()),("m",DecisionTreeClassifier(random_state=RS))]),
                  {"m__max_depth":[1,2,3,4,5,6,8,None]}, cv=cv, scoring="recall").fit(X_tr,y_tr)
rf = Pipeline([("sc",StandardScaler()),("m",RandomForestClassifier(n_estimators=300, random_state=RS))]).fit(X_tr,y_tr)
print("Best LR params:", lr.best_params_, "| Best tree params:", dt.best_params_)
models = {"Logistic Regression":lr.best_estimator_, "Decision Tree":dt.best_estimator_, "Random Forest":rf}

# ---------- 3. Evaluation ----------
rows=[]; probs={}
for n,m in models.items():
    pr = m.predict_proba(X_te)[:,1]; pd_ = (pr>=.5).astype(int); probs[n]=pr
    cvacc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
    rows.append(dict(Model=n, CV_accuracy=cvacc, Train_acc=m.score(X_tr,y_tr), Test_acc=accuracy_score(y_te,pd_),
         Precision=precision_score(y_te,pd_), Recall=recall_score(y_te,pd_), F1=f1_score(y_te,pd_), ROC_AUC=roc_auc_score(y_te,pr)))
R = pd.DataFrame(rows).set_index("Model"); print("\n", R.round(3).to_string())
for n,m in models.items():
    tn,fp,fn,tp = confusion_matrix(y_te,(probs[n]>=.5).astype(int)).ravel(); print(f"{n}: TN={tn} FP={fp} FN={fn} TP={tp}")

# bootstrap CI for the chosen model on the test set
rng = np.random.default_rng(RS); pl = (probs["Logistic Regression"]>=.5).astype(int); yt = y_te.values; ba,br=[],[]
for _ in range(2000):
    i = rng.integers(0,len(yt),len(yt)); ba.append((pl[i]==yt[i]).mean())
    pos = i[yt[i]==1]; br.append((pl[pos]==1).mean())
print("Bootstrap 95%% CI test accuracy: [%.3f, %.3f]; recall: [%.3f, %.3f]" % (*np.percentile(ba,[2.5,97.5]), *np.percentile(br,[2.5,97.5])))

# ---------- 4. Error analysis ----------
best = models["Logistic Regression"]; pr = probs["Logistic Regression"]
err = X_te.assign(true=y_te, prob_malignant=pr, pred=(pr>=.5).astype(int))
err = err[err.true!=err.pred]
print("\nMisclassified (LR):\n", err[["true","pred","prob_malignant","mean_radius","mean_concavity","worst_perimeter"]].round(3).to_string())
coef = pd.Series(best.named_steps["m"].coef_[0], index=X.columns).sort_values(); print("\nTop + coefs:\n", coef.tail(5).round(2), "\nTop - coefs:\n", coef.head(5).round(2))
for t in [0.5,0.4,0.3,0.2,0.1]:
    p_ = (pr>=t).astype(int); tn,fp,fn,tp = confusion_matrix(y_te,p_).ravel(); print(f"threshold {t}: FN={fn} FP={fp} recall={tp/(tp+fn):.3f} precision={tp/(tp+fp):.3f}")

# ---------- 5. Figures ----------
fig,ax = plt.subplots(1,3, figsize=(12,3.8))
for a,(n,m) in zip(ax,models.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_te,(probs[n]>=.5).astype(int)), display_labels=["Benign","Malignant"]).plot(ax=a, cmap="Blues", colorbar=False)
    a.set_title(n); a.grid(False)
plt.suptitle("Confusion matrices on the 114 unseen test patients", fontweight="bold"); plt.tight_layout(); plt.savefig("w4figs/m1_confusion.png", dpi=150); plt.close()

plt.figure(figsize=(6.2,5))
for n,c in zip(models,["#d62728","#1f77b4","#2ca02c"]):
    fpr,tpr,_ = roc_curve(y_te,probs[n]); plt.plot(fpr,tpr,color=c,lw=2,label=f"{n} (AUC = {R.loc[n,'ROC_AUC']:.3f})")
plt.plot([0,1],[0,1],"k--",lw=1,label="Random guess"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
plt.title("ROC curves", fontweight="bold"); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig("w4figs/m2_roc.png", dpi=150); plt.close()

M = R[["Test_acc","Precision","Recall","F1","ROC_AUC"]]
ax_ = M.T.plot(kind="bar", figsize=(8,4.5), color=["#d62728","#1f77b4","#2ca02c"], rot=0, width=.8)
ax_.set_ylim(.8,1.02); ax_.set_ylabel("Score"); ax_.set_title("Model comparison on the test set", fontweight="bold"); ax_.legend(loc="lower left")
plt.tight_layout(); plt.savefig("w4figs/m3_metrics.png", dpi=150); plt.close()

fig,ax = plt.subplots(1,2, figsize=(11,4.2))
depths = list(range(1,13)); tr_s=[];cv_s=[]
for d_ in depths:
    t_ = Pipeline([("sc",StandardScaler()),("m",DecisionTreeClassifier(max_depth=d_, random_state=RS))])
    tr_s.append(t_.fit(X_tr,y_tr).score(X_tr,y_tr)); cv_s.append(cross_val_score(t_,X_tr,y_tr,cv=cv).mean())
ax[0].plot(depths,tr_s,"o-",c="#ff7f0e",label="Training accuracy"); ax[0].plot(depths,cv_s,"o-",c="#1f77b4",label="Cross-validation accuracy")
ax[0].set_xlabel("Tree depth (model complexity)"); ax[0].set_ylabel("Accuracy"); ax[0].set_title("Decision tree: over-fitting as depth grows"); ax[0].legend()
sz,trs,vas = learning_curve(best, X_tr, y_tr, cv=cv, train_sizes=np.linspace(.1,1,8), scoring="accuracy", random_state=RS)
ax[1].plot(sz,trs.mean(1),"o-",c="#ff7f0e",label="Training accuracy"); ax[1].plot(sz,vas.mean(1),"o-",c="#1f77b4",label="Cross-validation accuracy")
ax[1].set_xlabel("Training set size"); ax[1].set_title("Logistic regression learning curve"); ax[1].legend()
plt.tight_layout(); plt.savefig("w4figs/m4_fit.png", dpi=150); plt.close()
print("tree depth table:", [(d_,round(a,3),round(b,3)) for d_,a,b in zip(depths,tr_s,cv_s)])
print("LR learning curve final: train %.3f cv %.3f" % (trs.mean(1)[-1], vas.mean(1)[-1]))

sel = pd.concat([coef.head(7), coef.tail(7)])
plt.figure(figsize=(7.5,5)); plt.barh(sel.index.str.replace("_"," "), sel.values, color=["#1f77b4" if v<0 else "#d62728" for v in sel.values])
plt.axvline(0,c="k",lw=.8); plt.xlabel("Coefficient (standardised features)"); plt.title("What drives the logistic regression?", fontweight="bold")
plt.tight_layout(); plt.savefig("w4figs/m5_coef.png", dpi=150); plt.close()

fig,ax = plt.subplots(figsize=(6.5,4.5))
p_,r_,t_ = precision_recall_curve(y_te, pr)
ax.plot(t_, p_[:-1], label="Precision", c="#1f77b4"); ax.plot(t_, r_[:-1], label="Recall", c="#d62728")
ax.axvline(.5, ls="--", c="grey"); ax.text(.51,.55,"default 0.5",color="grey")
ax.set_xlabel("Decision threshold for 'malignant'"); ax.set_ylabel("Score"); ax.set_title("Threshold trade-off: catching more cancers costs false alarms", fontsize=10, fontweight="bold"); ax.legend()
plt.tight_layout(); plt.savefig("w4figs/m6_threshold.png", dpi=150); plt.close()
R.round(4).to_csv("w4figs/model_results.csv")
