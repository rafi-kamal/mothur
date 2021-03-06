#ifndef MY_VERSION
#define MY_VERSION	"4.2"
#endif

STR_OPT(	input,					0)
STR_OPT(	query,					0)
STR_OPT(	db,						0)
STR_OPT(	sort,					0)
STR_OPT(	output,					0)
STR_OPT(	uc,						0)
STR_OPT(	clstr2uc,				0)
STR_OPT(	uc2clstr,				0)
STR_OPT(	uc2fasta,				0)
STR_OPT(	uc2fastax,				0)
STR_OPT(	mergesort,				0)
STR_OPT(	tmpdir,					".")
STR_OPT(	staralign,				0)
STR_OPT(	sortuc,					0)
STR_OPT(	blastout,				0)
STR_OPT(	blast6out,				0)
STR_OPT(	fastapairs,				0)
STR_OPT(	idchar,					"|")
STR_OPT(	diffchar,				" ")
STR_OPT(	uchime,					0)
STR_OPT(	gapopen,				0)
STR_OPT(	gapext,					0)
STR_OPT(	uhire,					0)
STR_OPT(	ids,					"99,98,95,90,85,80,70,50,35")
STR_OPT(	seeds,					0)
STR_OPT(	clump,					0)
STR_OPT(	clumpout,				0)
STR_OPT(	clump2fasta,			0)
STR_OPT(	clumpfasta,				0)
STR_OPT(	hireout,				0)
STR_OPT(	mergeclumps,			0)
STR_OPT(	alpha,					0)
STR_OPT(	hspalpha,				0)
STR_OPT(	probmx,					0)
STR_OPT(	matrix,					0)
STR_OPT(	tracestate,				0)
STR_OPT(	chainout,				0)
STR_OPT(	cluster,				0)
STR_OPT(	computekl,				0)
STR_OPT(	userout,				0)
STR_OPT(	userfields,				0)
STR_OPT(	seedsout,				0)
STR_OPT(	chainhits,				0)
STR_OPT(	findorfs,				0)
STR_OPT(	strand,					0)
STR_OPT(	getseqs,				0)
STR_OPT(	labels,					0)
STR_OPT(	doug,					0)
STR_OPT(	makeindex,				0)
STR_OPT(	indexstats,				0)
STR_OPT(	uchimeout,				0)
STR_OPT(	uchimealns,				0)
STR_OPT(	xframe,					0)
STR_OPT(	mkctest,				0)
STR_OPT(	allpairs,				0)
STR_OPT(	fastq2fasta,			0)
STR_OPT(	otusort,				0)
STR_OPT(	sparsedist,				0)
STR_OPT(	sparsedistparams,		0)
STR_OPT(	mcc,					0)
STR_OPT(	utax,					0)
STR_OPT(	simcl,					0)
STR_OPT(	absort,					0)
STR_OPT(	cc,						0)
STR_OPT(	uslink,					0)

UNS_OPT(	band,					16,			0,			UINT_MAX)
UNS_OPT(	minlen,					10,			1,			UINT_MAX)
UNS_OPT(	maxlen,					10000,		1,			UINT_MAX)
UNS_OPT(	w,						0,			1,			UINT_MAX)
UNS_OPT(	k,						0,			1,			UINT_MAX)
UNS_OPT(	stepwords,				8,			0,			UINT_MAX)
UNS_OPT(	maxaccepts,				1,			0,			UINT_MAX)
UNS_OPT(	maxrejects,				8,			0,			UINT_MAX)
UNS_OPT(	maxtargets,				0,			0,			UINT_MAX)
UNS_OPT(	minhsp,					32,			1,			UINT_MAX)
UNS_OPT(	bump,					50,			0,			100)
UNS_OPT(	rowlen,					64,			8,			UINT_MAX)
UNS_OPT(	idprefix,				0,			0,			UINT_MAX)
UNS_OPT(	idsuffix,				0,			0,			UINT_MAX)
UNS_OPT(	chunks,					4,			2,			UINT_MAX)
UNS_OPT(	minchunk,				64,			2,			UINT_MAX)
UNS_OPT(	maxclump,				1000,		1,			UINT_MAX)
UNS_OPT(	iddef,					0,			0,			UINT_MAX)
UNS_OPT(	mincodons,				20,			1,			UINT_MAX)
UNS_OPT(	maxovd,					8,			0,			UINT_MAX)
UNS_OPT(	max2,					40,			0,			UINT_MAX)
UNS_OPT(	querylen,				500,		0,			UINT_MAX)
UNS_OPT(	targetlen,				500,		0,			UINT_MAX)
UNS_OPT(	orfstyle,				(1+2+4),	0,			UINT_MAX)
UNS_OPT(	dbstep,					1,			1,			UINT_MAX)
UNS_OPT(	randseed,				1,			0,			UINT_MAX)
UNS_OPT(	maxp,					2,			2,			UINT_MAX)
UNS_OPT(	idsmoothwindow,			32,			1,			UINT_MAX)
UNS_OPT(	mindiffs,				3,			1,			UINT_MAX)
UNS_OPT(	maxspan1,				24,			1,			UINT_MAX)
UNS_OPT(	maxspan2,				24,			1,			UINT_MAX)
UNS_OPT(	minorfcov,				16,			1,			UINT_MAX)
UNS_OPT(	hashsize,				4195879,	1,			UINT_MAX)
UNS_OPT(	maxpoly,				0,			0,			UINT_MAX)
UNS_OPT(	droppct,				50,			0,			100)
UNS_OPT(	secs,					10,			0,			UINT_MAX)
UNS_OPT(	maxqgap,				0,			0,			UINT_MAX)
UNS_OPT(	maxtgap,				0,			0,			UINT_MAX)

INT_OPT(	frame,					0,			-3,			+3)

TOG_OPT(	trace,					false)
TOG_OPT(	logmemgrows,			false)
TOG_OPT(	trunclabels,			false)
TOG_OPT(	verbose,				false)
TOG_OPT(	wordcountreject,		true)
TOG_OPT(	rev,					false)
TOG_OPT(	output_rejects,			false)
TOG_OPT(	blast_termgaps,			false)
TOG_OPT(	fastalign,				true)
TOG_OPT(	flushuc,				false)
TOG_OPT(	stable_sort,			false)
TOG_OPT(	minus_frames,			true)
TOG_OPT(	usort,					true)
TOG_OPT(	nb,						false)
TOG_OPT(	twohit,					true)
TOG_OPT(	ssort,					false)
TOG_OPT(	log_query,				false)
TOG_OPT(	log_hothits,			false)
TOG_OPT(	logwordstats,			false)
TOG_OPT(	ucl,					false)
TOG_OPT(	skipgaps2,				true)
TOG_OPT(	skipgaps,				true)
TOG_OPT(	denovo,					false)
TOG_OPT(	cartoon_orfs,			false)
TOG_OPT(	label_ab,				false)
TOG_OPT(	wordweight,				false)
TOG_OPT(	isort,					false)
TOG_OPT(	selfid,					false)
TOG_OPT(	leftjust,				false)
TOG_OPT(	rightjust,				false)

FLT_OPT(	id,						0.0,		0.0,		1.0)
FLT_OPT(	weak_id,				0.0,		0.0,		1.0)
FLT_OPT(	match,					1.0,		0.0,		FLT_MAX)
FLT_OPT(	mismatch,				-2.0,		0.0,		FLT_MAX)
FLT_OPT(	split,					1000.0,		1.0,		FLT_MAX)
FLT_OPT(	evalue,					10.0,		0.0,		FLT_MAX)
FLT_OPT(	weak_evalue,			10.0,		0.0,		FLT_MAX)
FLT_OPT(	evalue_g,				10.0,		0.0,		FLT_MAX)
FLT_OPT(	chain_evalue,			10.0,		0.0,		FLT_MAX)
FLT_OPT(	xdrop_u,				16.0,		0.0,		FLT_MAX)
FLT_OPT(	xdrop_g,				32.0,		0.0,		FLT_MAX)
FLT_OPT(	xdrop_ug,				16.0,		0.0,		FLT_MAX)
FLT_OPT(	xdrop_nw,				16.0,		0.0,		FLT_MAX)
FLT_OPT(	ka_gapped_lambda,		0.0,		0.0,		FLT_MAX)
FLT_OPT(	ka_ungapped_lambda,		0.0,		0.0,		FLT_MAX)
FLT_OPT(	ka_gapped_k,			0.0,		0.0,		FLT_MAX)
FLT_OPT(	ka_ungapped_k,			0.0,		0.0,		FLT_MAX)
FLT_OPT(	ka_dbsize,				0.0,		0.0,		FLT_MAX)
FLT_OPT(	chain_targetfract,		0.0,		0.0,		1.0)
FLT_OPT(	targetfract,			0.0,		0.0,		1.0)
FLT_OPT(	queryfract,				0.0,		0.0,		1.0)
FLT_OPT(	fspenalty,				16.0,		0.0,		FLT_MAX)
FLT_OPT(	sspenalty,				20.0,		0.0,		FLT_MAX)
FLT_OPT(	seedt1,					13.0,		0.0,		FLT_MAX)
FLT_OPT(	seedt2,					11.0,		0.0,		FLT_MAX)
FLT_OPT(	lopen,					11.0,		0.0,		FLT_MAX)
FLT_OPT(	lext,					1.0,		0.0,		FLT_MAX)
FLT_OPT(	minh,					0.3,		0.0,		FLT_MAX)
FLT_OPT(	xn,						8.0,		0.0,		FLT_MAX)
FLT_OPT(	dn,						1.4,		0.0,		FLT_MAX)
FLT_OPT(	xa,						1.0,		0.0,		FLT_MAX)
FLT_OPT(	mindiv,					0.5,		0.0,		100.0)
FLT_OPT(	abskew,					2,			0.0,		100.0)
FLT_OPT(	abx,					8.0,		0.0,		100.0)
FLT_OPT(	minspanratio1,			0.7,		0.0,		1.0)
FLT_OPT(	minspanratio2,			0.7,		0.0,		1.0)

FLAG_OPT(	usersort)
FLAG_OPT(	exact)
FLAG_OPT(	optimal)
FLAG_OPT(	self)
FLAG_OPT(	ungapped)
FLAG_OPT(	global)
FLAG_OPT(	local)
FLAG_OPT(	xlat)
FLAG_OPT(	realign)
FLAG_OPT(	hash)
FLAG_OPT(	derep)
