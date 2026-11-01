# Phase 0: evidence for the reasoning model

Checked 1 November 2026. Rule: a direction is enforced only if a source outside this bank's own data supports it. Read level: (a) full text or relevant sections read, (b) abstract only.

## Result

**One direction is supported: a previous campaign success raises uptake (moderate, by analogy).** Every other feature is left free inside its factor. The structure of the models therefore comes mainly from the factor grouping, not from directions.

| Feature | Direction | Evidence | Why |
| --- | --- | --- | --- |
| `prev_success` | Increasing | Moderate | Past response and purchase frequency predict future response in direct marketing and donor data (Viaene et al. 2001; van Diepen et al. 2009; Fader et al. 2005). Analogy: not bank telemarketing |
| `days_since_contact` | None | Weak | Recency of contact is not recency of purchase; response recency has an inverted-U effect in donor data (van Diepen et al.) |
| `previous` | None | Contradictory | More contacts lower response (van Diepen et al.); more purchases raise it (Viaene; Fader). The feature counts contacts of responders and non-responders alike |
| Failure vs never contacted | None | Weak | Fatigue evidence is indirect; no study compares the two |
| `contact` | None | Weak | Only US survey-cooperation evidence (AAPOR 2010), which does not transfer to bank clients in Portugal |
| `default` | None | Weak | "yes" has 3 clients in training; "unknown" kept as its own category |
| `housing`, `loan` | None | Weak | Indebtedness rises with income in Portugal (Costa and Farinha 2012); indebted Portuguese households still hold relatively many liquid assets (Ampudia et al. 2016) |
| `euribor3m` | None | Contradictory | Lower short rates move savers out of term deposits in theory and euro-area data (Drechsler et al. 2017; Stinglhamber et al. 2011); this bank's data shows the opposite (Moro et al. 2014, not external); Portuguese deposit rates decoupled from Euribor from 2008 (Esteves and Pinheiro 2020) |
| `emp.var.rate`, `nr.employed`, `cons.conf.idx` | None | Weak for term deposits | Uncertainty raises aggregate saving (Mody et al. 2012), but crisis savings went into liquid accounts, not term deposits (Stinglhamber et al.) |
| `cons.price.idx` | None | None found | |
| Life stage | None | Weak | Sensitive; fitted with and without |

## What the original study says

Moro, Cortez and Rita (2014), read in full: their final model used 22 of 150 attributes from 2008 to 2013. Previous outcome, days since contact, number of contacts and contact type were **not** among the 22. The strongest inputs were Euribor, call direction (inbound or outbound), agent experience, and the offered rate minus the national average. The last three are not in the public file. The UCI market indicators therefore partly stand in for the bank's own offers and campaign strategy over time.

## Expected costs, from the literature

- **Directions:** Koklev (2025, read in full) measured the cost of monotone constraints in LightGBM on five credit datasets: −0.0% to 2.9% relative AUC, near zero on large data, about 1.2% on a 21,000-row dataset with 39% of features constrained. Here one binary feature of 20 is constrained, so the expected cost is about zero.
- **Factor grouping:** no study measures it. Main-effects-only additive models trail tuned boosting by 0 to 2 AUC points on medium tabular data, and adding pairwise interactions closes most of that gap (Lou et al. 2013; Chang et al. 2021; Chang et al. 2022). Factor grouping allows any interaction inside a factor and none across, so its cost should sit between those cases: roughly 0.002 to 0.010 absolute AUC, possibly up to 0.02. This is an inference, not a measurement.

## Sources

- Moro, Cortez and Rita (2014), Decision Support Systems 62. [ISCTE preprint](https://repositorio.iscte-iul.pt/bitstreams/51d70c26-0646-4f4a-a454-cbd07136018d/download) (a)
- Moro, Laureano and Cortez (2011), ESM 2011. [UMinho](https://repositorium.uminho.pt/bitstreams/8094aa93-2392-46f1-a1c3-d7bd98bc2a83/download) (a)
- Viaene et al. (2001), Int. J. Intelligent Systems 16. [KU Leuven report](https://ftp.esat.kuleuven.be/pub/stadius/ida/reports/00-144.PDF) (a)
- van Diepen, Donkers and Franses (2009), JMR 46(1). [Working paper](https://repub.eur.nl/pub/7948/ERS-2006-050-MKT.pdf) (a)
- Fader, Hardie and Lee (2005), JMR 42(4). [Preprint](http://www.brucehardie.com/papers/rfm_clv_2005-02-16.pdf) (a)
- AAPOR Cell Phone Task Force (2010). [Report](https://aapor.org/wp-content/uploads/2023/01/2010AAPORCellPhoneTFReport.pdf) (a)
- Costa and Farinha (2012), Banco de Portugal OP 1/2012. [PDF](https://www.bportugal.pt/sites/default/files/anexos/papers/op201201.pdf) (a)
- Ampudia, van Vlokhoven and Zochowski (2016), ECB WP 1737. [PDF](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1737.en.pdf) (a)
- Drechsler, Savov and Schnabl (2017), QJE. [NBER WP 22152](https://www.nber.org/system/files/working_papers/w22152/w22152.pdf) (a)
- Stinglhamber, Van Nieuwenhuyze and Zachary (2011), NBB Economic Review. [PDF](https://www.nbb.be/doc/ts/publications/economicreview/2011/ecorevii2011_h5.pdf) (a)
- Esteves and Pinheiro (2020), Banco de Portugal Economic Studies. [PDF](https://www.bportugal.pt/sites/default/files/anexos/papers/re202010_en.pdf) (a)
- Mody, Ohnsorge and Sandri (2012), IMF WP/12/42. [PDF](https://www.imf.org/external/pubs/ft/wp/2012/wp1242.pdf) (a)
- Koklev (2025), What's the price of monotonicity? [arXiv 2512.17945](https://arxiv.org/abs/2512.17945) (a)
- Lou, Caruana, Gehrke and Hooker (2013), KDD. [PDF](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf) (a)
- Chang, Tan, Lengerich, Goldenberg and Caruana (2021), KDD. [arXiv 2006.06466](https://arxiv.org/abs/2006.06466) (a)
- Chang, Caruana and Goldenberg (2022), ICLR. [arXiv 2106.01613](https://arxiv.org/abs/2106.01613) (a)
- Goyal, Dumancic and Blockeel (2020). [arXiv 2007.05758](https://arxiv.org/abs/2007.05758) (a)
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) (a)
