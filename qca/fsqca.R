library(QCA)
d <- read.csv("../data/processed/qca_calibrated.csv", row.names="startup_id")
conds <- c("PA_fuzzy","SF_fuzzy","TGB_f","TGT_f","TGR_f","TEAM_f","PLAT_f","B2B_f")

cat("\n=== NECESSITY (ladder outcome) ===\n")
print(superSubset(d, outcome="OUTCOME_ladder", conditions=conds,
      relation="necessity", incl.cut=0.9))

cat("\n=== NECESSITY (accept outcome) ===\n")
print(superSubset(d, outcome="OUTCOME_accept", conditions=conds,
      relation="necessity", incl.cut=0.9))

cat("\n=== TRUTH TABLE (accept outcome) ===\n")
tt <- truthTable(d, outcome="OUTCOME_accept", conditions=conds,
                 incl.cut=0.8, n.cut=1, show.cases=TRUE, sort.by="incl")
print(tt)

cat("\n=== MINIMIZATION (intermediate solution) ===\n")
print(minimize(tt, details=TRUE, include="?"))
