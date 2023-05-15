# Supporting information readme for "Beyond strain release: Delocalization-enabled organic reactivity"

## Alistair J. Sterling, Russell C. Smith, Edward A. Anderson & Fernanda Duarte

*ChemRxiv* **2021**: https://doi.org/10.26434/chemrxiv-2021-n0xm9-v2

This readme contains information on the supporting data that accompanies the manuscript described above. The file structure is depicted, followed by a description of the contents of each directory, subdirectory and file.

### File structure

	CartesianCoordinates 	> 	CCl3_addition		>	111P
								>	BC210P
								> 	BCB

				>	CH3_addition		>	111P
								>	211P
								>	221P
								>	222P	
								>	311P
								>	BC210P
								>	BC220H
								>	BC310H
								>	BCB
								>	cyclobutane
								>	cyclopropane
								>	ethane

				>	Cycloaddition		>	cyclodecyne
								>	cycloheptyne
								>	cyclononyne
								>	cyclooctyne
								>	dibenzocyclooctyne
								>	distal_monobenzocyclooctyne
								>	F2-cyclooctyne
								>	hex-3-yne
								>	monobenzocyclooctyne
								>	NS-cyclooctyne

				>	NH2_addition		>	111P
								>	211P
								>	221P
								>	222P	
								>	311P
								>	BC210P
								>	BC220H
								>	BC310H
								>	BCB
								>	cyclobutane
								>	cyclopropane
								>	ethane

				>	Strain_database

	pdfs			>	Plot1.pdf
					Plot2.pdf
					Plot3.pdf
					Plot4.pdf
					Plot5.pdf
					Plot6.pdf
					Plot7.pdf
					Plot8.pdf
					Plot9.pdf
					Plot10.pdf
					Plot11.pdf
					Plot12.pdf
					Plot13.pdf
					Plot14.pdf
					Plot15.pdf
					Plot16.pdf
					Plot17.pdf
					Plot18.pdf
					Plot19.pdf
					Plot20.pdf

	amide_data.csv
	cycloaddition_data.csv
	Hoz_data.csv
	hydrocarbons_data.csv
	SRE.csv

	SRE.xlsx
	energies.xlsx

	main.py
	
	Supporting_Information.pdf


### File descriptions

#### Supporting_Information.pdf

"[Supporting_Information.pdf](https://github.com/duartegroup/strain-delocalisation/blob/main/Supporting_Information.pdf)" contains a detailed description of the methodologies employed in this work, as well as supplementary figures, tables, and references.

#### SRE.csv

"SRE.csv" is a database containing strain release energies (kcal/mol) of unique bonds of 35 molecules (81 total data points). 
The columns contain the following information:

*Column:*
+ 0 - Name of strained molecule
+ 1 - Name of unstrained product molecule
+ 2 - SMILES string representation of balanced reaction to obtain strain release energy, where dots (".") are used to separate individual molecules, and a double chevron (">>") indicates the reaction arrow
+ 3 - Bond type designation, when there are multiple unique bond types in a given molecule
+ 4 - Strain release energy (SRE, kcal/mol)
+ 5 - Delocalisation value for the breaking bond (2–Nocc, *e*)

N.B. Strain release energy is defined as the change in enthalpy (∆H in kcal/mol) at the DLPNO-CCSD(T)/def2-QZVPP//B2PLYP-D3BJ/def2-TZVP level, and 2–Nocc values (in *e*) at are obtained from the B2PLYP/def2-TZVP relaxed density.

#### main.py

"main.py" is a script to plot all figures. Multiple linear regression is carried out using the [Scikit-learn](https://scikit-learn.org/stable/) Python package, and plotting is done using [Matplotlib](https://matplotlib.org). "main.py" can either be run interactively, or to generate all plots, run:

>	for i in {1..20}; do echo $i | python main.py -v; done

Plots are generated inside the “pdfs” directory. Each plot contains the following:
+ [Plot 1](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot1.pdf): y = ∆H‡, x0 = ∆H0, x1 = 2-Nocc (CH3• addition)
+ [Plot 2](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot2.pdf): y = ∆H‡, x0 = ∆H0, x1 = D/D_0 (CH3• addition)
+ [Plot 3](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot3.pdf): y = ∆H‡, x0 = ∆H0, x1 = n3 (CH3• addition)
+ [Plot 4](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot4.pdf): y = ∆H‡, x0 = ∆H0 (CH3• addition); ∆Hr vs ∆H‡
+ [Plot 5](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot5.pdf): y = ∆H‡, x0 = ∆H0 (CH3• addition); ∆H‡ calc vs pred
+ [Plot 6](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot6.pdf): y = ∆H‡, x0 = ∆H0, x1 = 2-Nocc, x2 = (∆H0)^2 (CH3• addition)
+ [Plot 7](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot7.pdf): y = ∆H‡, x0 = ∆H0, x1 = D/D_0, x2 = (∆H0)^2 (CH3• addition)
+ [Plot 8](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot8.pdf): y = ∆H‡, x0 = (∆r‡)^2 (CH3• addition)
+ [Plot 9](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot9.pdf): y = (∆r‡)^2, x0 = ∆H0 (CH3• addition)
+ [Plot 10](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot10.pdf): y = ∆r‡, x0 = ∆H0, x1 = 2-Nocc (CH3• addition)
+ [Plot 11](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot11.pdf): y = ∆r‡, x0 = ∆H0, x1 = D/D_0 (CH3• addition)
+ [Plot 12](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot12.pdf): y = ∆H‡, x0 = ∆H0, x1 = n3 (NH2- addition)
+ [Plot 13](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot13.pdf): y = Marcus Ea error, x0 = n3 (heterosubstitution)
+ [Plot 14](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot14.pdf): y = ∆H‡, x0 = ∆H0, x1 = (∆H0)^2 (CH3• addition)
+ [Plot 15](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot15.pdf): y = ∆H‡, x0 = ∆H0, x1 = 2-Nocc (NH2- addition)
+ [Plot 16](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot16.pdf): y = ∆H‡, x0 = ∆H0 (cycloaddition); ∆H‡ calc vs pred
+ [Plot 17](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot17.pdf): y = ∆H‡, x0 = ∆H0, x1 = mean 2-Nocc (cycloaddition)
+ [Plot 18](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot18.pdf): y = ∆H‡, x0 = ∆H0, x1 = E_HOMO (CH3• addition)
+ [Plot 19](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot19.pdf): y = ∆H‡, x0 = ∆H0, x1 = E_LUMO (CH3• addition)
+ [Plot 20](https://github.com/duartegroup/strain-delocalisation/tree/main/pdfs/Plot20.pdf): y = ∆H‡, x0 = ∆H0, x1 = ∆E_HOMO-LUMO (CH3• addition)

#### Cartesian coordinates

The directory [CartesianCoordinates](https://github.com/duartegroup/strain-delocalisation/tree/main/CartesianCoordinates) contains all xyz files generated in this study, split into subdirectories ([CCl3_addition](https://github.com/duartegroup/strain-delocalisation/tree/main/CartesianCoordinates/CCl3_addition), [CH3_addition](https://github.com/duartegroup/strain-delocalisation/tree/main/CartesianCoordinates/CH3_addition), [Cycloaddition](https://github.com/duartegroup/strain-delocalisation/tree/main/CartesianCoordinates/Cycloaddition), [NH2_addition](https://github.com/duartegroup/strain-delocalisation/tree/main/CartesianCoordinates/NH2_addition), [Strain_database](https://github.com/duartegroup/strain-delocalisation/tree/main/CartesianCoordinates/Strain_database)). The numbering of molecules in “[Strain_database](https://github.com/duartegroup/strain-delocalisation/tree/main/CartesianCoordinates/Strain_database)” follows that defined in “[smiles.csv](https://github.com/duartegroup/strain-delocalisation/blob/main/CartesianCoordinates/Strain_database/smiles.csv)” contained within the same subdirectory. Geometries were optimised at the levels of theory described in the corresponding tabs in “[energies.xlsx](https://github.com/duartegroup/strain-delocalisation/blob/main/energies.xlsx)”, the [Supporting Information](https://github.com/duartegroup/strain-delocalisation/blob/main/Supporting_Information.pdf), and below in this readme.


#### hydrocarbons_data.csv

The format of "[hydrocarbons_data.csv](https://github.com/duartegroup/strain-delocalisation/blob/main/hydrocarbons_data.csv)" is as follows:

> *Column:*
> + 0 - Molecule identifier
> + 1 - TS enthalpy / kcal/mol
> + 2 - reaction enthalpy / kcal/mol
> + 3 - 1–ELF (B2PLYP/def2-TZVP)
> + 4 - 2–N_{occ} (B2PLYP/def2-TZVP, relaxed density)
> + 5 - Number of three membered rings appended to breaking bond
> + 6 - SM breaking C-C / Angstrom
> + 7 - TS breaking C-C / Angstrom
> + 8 - HOMO energy associated with breaking bond / Ha
> + 9 - LUMO energy associated with breaking bond / Ha
>
> *Row:*
> + 1 - ethane, **A**
> + 2 - cyclopropane, **B**
> + 3 - cyclobutane, **C**
> + 4 - bicyclo[1.1.0]butane, BCB, **D**
> + 5 - bicyclo[2.1.0]pentane, BC210P, **E**
> + 6 - bicyclo[3.1.0]hexane, BC310H, **F**
> + 7 - bicyclo[2.2.0]hexane, BC220H, **G**
> + 8 - [1.1.1]propellane, 111P, **H**
> + 9 - [2.1.1]propellane, 211P, **I**
> + 10 - [3.1.1]propellane, 311P, **J**
> + 11 - [2.2.1]propellane, 221P, **K**
> + 12 - [2.2.2]propellane, 222P, **L**


#### amide_data.csv

The format of "[amide_data.csv](https://github.com/duartegroup/strain-delocalisation/blob/main/amide_data.csv)" is as follows:

> *Column:*
> + 0 - Molecule identifier
> + 1 - TS enthalpy / kcal/mol
> + 2 - reaction enthalpy / kcal/mol
> + 3 - Number of three membered rings appended to breaking bond
> + 4 - 2-Nocc (B2PLYP/def2-TZVP, relaxed density)
>
> *Row:*
> + 1 - ethane, **A**
> + 2 - cyclopropane, **B**
> + 3 - cyclobutane, **C**
> + 4 - bicyclo[1.1.0]butane, BCB, **D**
> + 5 - bicyclo[2.1.0]pentane, BC210P, **E**
> + 6 - bicyclo[3.1.0]hexane, BC310H, **F**
> + 7 - bicyclo[2.2.0]hexane, BC220H, **G**
> + 8 - [1.1.1]propellane, 111P, **H**
> + 9 - [2.1.1]propellane, 211P, **I**
> + 10 - [3.1.1]propellane, 311P, **J**
> + 11 - [2.2.1]propellane, 221P, **K**
> + 12 - [2.2.2]propellane, 222P, **L**


#### Hoz_data.csv

The format of "[Hoz_data.csv](https://github.com/duartegroup/strain-delocalisation/blob/main/Hoz_data.csv)" is as follows:

> *Column:*
> + 0 - Reaction type (anionic or radical)
> + 1 - Reactant 
> + 2 - Substrate
> + 3 - Reaction energy / kcal/mol
> + 4 - TS energy (activation energy) / kcal/mol
> + 5 - Marcus intrinsic activation barrier / kcal/mol
> + 6 - Marcus predicted activation barrier / kcal/mol
> + 7 - Difference between calculated and predicted (Marcus) activation energies / kcal/mol
> + 8 - 2–N_{occ} (B2PLYP/def2-TZVP, relaxed density)
> + 9 - Number of three membered rings appended to breaking bond
> + 10 - Leaving group heteroatom type


#### cycloaddition_data.csv

The format of "[cycloaddition_data.csv](https://github.com/duartegroup/strain-delocalisation/blob/main/cycloaddition_data.csv)" is as follows:

> *Column:*
> + 0 - Alkyne reactant (syn / anti denotes stereochemistry of the product)
> + 1 - TS enthalpy (kcal/mol)
> + 2 - Reaction enthalpy (kcal/mol)
> + 3 - Mean 2–N_{occ} value (from the two 2–N_{occ} values obtained for the two alkyne pi bonds)


#### energies.xlsx

The format of "[energies.xlsx](https://github.com/duartegroup/strain-delocalisation/blob/main/energies.xlsx)" is as follows:

*Tab 1*

> TS and reaction energies for the addition of CH3 radical to H12 hydrocarbons. Raw energies in Ha, energy differences in kcal/mol.
> 
> **Optimisation:** B2PLYP D3BJ def2-TZVP Grid5 FinalGrid6 GridX6 def2/J def2/TZVP/C RIJCOSX TightOpt NumFreq  
> **Single point:** DLPNO-CCSD(T) AutoAux def2-QZVPP TightSCF PAL4 Grid6 FinalGrid6 GridX6 TightPNO RIJCOSX

*Tab 2*

> TS and reaction energies for the addition of NH2– to H12 hydrocarbons. Raw energies in Ha, energy differences in kcal/mol.
> 
> **Optimisation:** B2PLYP D3BJ def2-TZVP Grid5 FinalGrid6 GridX6 def2/J def2/TZVP/C RIJCOSX TightOpt NumFreq SMD(THF) *NB: ma-def2-TZVP basis set added to nucleophilic N*  
> **Single point:** DLPNO-CCSD(T) AutoAux ma-def2-QZVPP TightSCF PAL4 Grid6 FinalGrid6 GridX6 TightPNO SMD(THF)

*Tab 3*

> VdW complex, TS and reaction energies for the addition of CCl3 radical to [1.1.1]propellane, bicyclo[1.1.0]butane and bicyclo[2.1.0]pentane. Raw energies in Ha, energy differences in kcal/mol.
>
> **Optimisation:** B2PLYP D3BJ def2-TZVP Grid5 FinalGrid6 GridX6 def2/J def2/TZVP/C RIJCOSX TightOpt NumFreq  
> **Single point:** DLPNO-CCSD(T) AutoAux def2-QZVPP TightSCF PAL4 Grid6 FinalGrid6 GridX6 TightPNO RIJCOSX

*Tab 4*

> Data extracted from 10.1021/jo001412t and 10.1039/b314869f in kcal/mol

*Tab 5*

> 2–N_{occ} and n3 values for molecules studied in this paper

*Tab 6*

> Balanced hydrogen transfer reactions for the dataset in Figure 5c. Raw energies in Ha, energy differences in kcal/mol.
> 
> **Optimisation:** B2PLYP D3BJ def2-TZVP Grid6 FinalGrid6 GridX6 def2/J def2/TZVP/C RIJCOSX TightOpt NumFreq  
> **Single point:** DLPNO-CCSD(T) AutoAux def2-QZVPP TightSCF PAL4 Grid6 FinalGrid6 GridX6 TightPNO RIJCOSX

*Tab 7*

> Cycloaddtion TSs and minima for the addition of methyl azide to a selection of alkynes. Raw energies in Ha, energy differences in kcal/mol. Mean 2–N_{occ} values obtained from the two 2–N_{occ} values corresponding to each of the pi bonds of the alkyne.
> 
> **Optimisation:** B2PLYP D3BJ def2-TZVP Grid6 FinalGrid6 GridX6 def2/J def2/TZVP/C RIJCOSX TightOpt NumFreq
