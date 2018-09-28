#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace std;

class OMPRewriter {
	using RewriteOptions = Rewriter::RewriteOptions;
	using buffer_iterator = Rewriter::buffer_iterator;
	using const_buffer_iterator = Rewriter::const_buffer_iterator;

	public:
	OMPRewriter() : Rewriter_() {}
	OMPRewriter(SourceManager &SM, const LangOptions &LO) : Rewriter_(SM, LO) {}


	void setSourceMgr(SourceManager &SM, const LangOptions &LO) {
		Rewriter_.setSourceMgr(SM, LO);
	}

	const LangOptions& getLangOpts() const {
		return Rewriter_.getLangOpts();
	}
	int getRangeSize(SourceRange Range, RewriteOptions opts=RewriteOptions()) const {
		return Rewriter_.getRangeSize(Range, opts);
	}
	int getRangeSize(const CharSourceRange &Range, RewriteOptions opts=RewriteOptions()) const {
		return Rewriter_.getRangeSize(Range, opts);
	}
	std::string	getRewrittenText(SourceRange Range) const {
		return Rewriter_.getRewrittenText(Range);
	}
	bool InsertText(SourceLocation Loc, StringRef Str, bool InsertAfter=true, bool indentNewLines=false) {
		return Rewriter_.InsertText(Loc, Str, InsertAfter, indentNewLines);
	}
	bool InsertTextAfter (SourceLocation Loc, StringRef Str) {
		return Rewriter_.InsertTextAfter(Loc, Str);
	}
	bool InsertTextAfterToken (SourceLocation Loc, StringRef Str) {
		return Rewriter_.InsertTextAfterToken(Loc, Str); 
	}
	bool InsertTextBefore (SourceLocation Loc, StringRef Str) {
		return Rewriter_.InsertTextBefore(Loc, Str);
	}
	bool RemoveText(SourceLocation Start, unsigned Length, RewriteOptions opts=RewriteOptions()) {
		return Rewriter_.RemoveText(Start, Length, opts);
	}
	bool RemoveText (CharSourceRange range, RewriteOptions opts=RewriteOptions()) {
		return Rewriter_.RemoveText(range, opts);
	}
	bool RemoveText (SourceRange range, RewriteOptions opts=RewriteOptions()) {
		return Rewriter_.RemoveText(range, opts);
	}
	bool ReplaceText (SourceLocation Start, unsigned OrigLength, StringRef NewStr) {
		return Rewriter_.ReplaceText(Start, OrigLength, NewStr);
	}
	bool ReplaceText (SourceRange range, StringRef NewStr) {
		return Rewriter_.ReplaceText(range, NewStr);
	}
	bool ReplaceText (SourceRange range, SourceRange replacementRange) {
		return Rewriter_.ReplaceText(range, replacementRange);
	}
	bool IncreaseIndentation (CharSourceRange range, SourceLocation parentIndent) {
		return Rewriter_.IncreaseIndentation(range, parentIndent);
	}
	bool IncreaseIndentation (SourceRange range, SourceLocation parentIndent) {
		return Rewriter_.IncreaseIndentation(range, parentIndent);
	}
	RewriteBuffer& getEditBuffer (FileID FID) {
		return Rewriter_.getEditBuffer(FID);
	}
	const RewriteBuffer* getRewriteBufferFor (FileID FID) const {
		return Rewriter_.getRewriteBufferFor(FID);
	}
	buffer_iterator buffer_begin () {
		return Rewriter_.buffer_begin();
	}
	buffer_iterator 	buffer_end () {
		return Rewriter_.buffer_end();
	}
	const_buffer_iterator 	buffer_begin () const {
		return Rewriter_.buffer_begin();
	}
	const_buffer_iterator 	buffer_end () const {
		return Rewriter_.buffer_end();
	}
	bool overwriteChangedFiles () {
		return Rewriter_.overwriteChangedFiles();
	}

	unsigned getLocationOffsetAndFileID(SourceLocation Loc, FileID &FID) {
		assert(Loc.isValid() && "Invalid location");
		std::pair<FileID, unsigned> V = Rewriter_.getSourceMgr().getDecomposedLoc(Loc);
		FID = V.first;
		return V.second;
	}

	bool InsertBeforeToken(SourceLocation Loc, StringRef Str) {
		if (!Rewriter::isRewritable(Loc)) return true;
		FileID FID;
		unsigned StartOffs = getLocationOffsetAndFileID(Loc, FID);
		Rewriter::RewriteOptions rangeOpts;
		rangeOpts.IncludeInsertsAtBeginOfRange = false;
		StartOffs += getRangeSize(SourceRange(Loc, Loc), rangeOpts);
		getEditBuffer(FID).InsertText(StartOffs, Str, /*InsertAfter*/false);
		return false;
	}

	SourceRange getPragmaRange(OMPExecutableDirective *D) {
		SourceManager &SrcMgr = Rewriter_.getSourceMgr();
		FileID FID = SrcMgr.getFileID(D->getBeginLoc());
		unsigned BeginLineNo = SrcMgr.getPresumedLineNumber(D->getBeginLoc());
		SourceLocation BeginLoc = SrcMgr.translateLineCol(FID, BeginLineNo, 1); // first column!
		SourceLocation EndLoc = D->getEndLoc();

		return SourceRange(BeginLoc, EndLoc);
	}

	void CommentExecutableDirective(OMPExecutableDirective *D) {
		// comment out pragma
		SourceManager &SrcMgr = Rewriter_.getSourceMgr();
		FileID FID = SrcMgr.getFileID(D->getBeginLoc());
		unsigned BeginLineNo = SrcMgr.getPresumedLineNumber(D->getBeginLoc());
		unsigned EndLineNo = SrcMgr.getPresumedLineNumber(D->getEndLoc());
		for (unsigned LineNo = BeginLineNo; LineNo <= EndLineNo; LineNo++) {
			SourceLocation BeginLoc = SrcMgr.translateLineCol(FID, LineNo, 1);
			InsertTextAfter(BeginLoc, "//");
		}
	}


	void TransformExecutableDirectiveOfDistribute(OMPExecutableDirective *D) {
		assert(isa<OMPLoopDirective>(D) && "TransformExecutableDirectiveOfDistribute Failed");

		CapturedStmt *CaptStmt = D->getInnermostCapturedStmt();
		CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
		PrintingPolicy Policy(Rewriter_.getLangOpts());
		Stmt *Body = CaptDecl->getBody();
		SourceRange PragmaRange = getPragmaRange(D);
		SourceRange WholeRange = D->getInnermostCapturedStmt()->getCapturedStmt()->getSourceRange();

		OMPLoopDirective *LoopDirective = dyn_cast<OMPLoopDirective>(D);

		Rewriter::RewriteOptions removeOpts;
		removeOpts.IncludeInsertsAtBeginOfRange = false;
		removeOpts.IncludeInsertsAtEndOfRange = false;



		
		SourceLocation KernelStartLoc = Rewriter_.getSourceMgr().getLocForStartOfFile(KernelFileID_);
		RemoveText(SourceRange(Body->getBeginLoc(), Body->getEndLoc()), removeOpts);
		// FIXME
		// Insert Kernel Call;
		InsertTextAfter(Body->getBeginLoc(), "KernelCall<<< 1, 2, 3, 4 >>>();\n");
		// Insert Kernel function;
		InsertTextAfter(KernelStartLoc, "KernelCall() {}");



		//			OMPCollapseClause *CollapseClause = getSingleClause<OMPCollapseClause>();
		//			Expr *NumForLoops;
		//			if (CollapseClause == nullptr) {
		//			} else {
		//				if ((NumForLoops = CollapseClause->getNumForLoops()) == nullptr) {
		//				} else {
		//				}
		//			}
		//
		//			OMPDistScheduleClause *ScheduleClause = getSingleClause<OMPDistScheduleClause>();
		//
		//			OpenMPDistScheduleClauseKind ScheduleKind;
		//			Expr *ChunkSize;
		//			if (ScheduleClause == nullptr) {
		//				// Default Scheduling is implementation specific.
		//				// We choose here 'static' scheduling as default.
		//				ScheduleKind = OMPC_DIST_SCHEDULE_static;
		//			} else {
		//				ScheduleKind = ScheduleClause->getDistScheduleKind();
		//				if ((ChunkSize = ScheduleClause->getChunkSize()) == nullptr) {
		//					// Approximately equal in size.
		//					ChunkSize = 
		//				}
		//			}
	}

	void TransformExecutableDirectiveOfTeams (OMPExecutableDirective *D) {
		CapturedStmt *CaptStmt = D->getInnermostCapturedStmt();
		CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
		PrintingPolicy Policy(Rewriter_.getLangOpts());
		Stmt *Body = CaptDecl->getBody();
		SourceRange PragmaRange = getPragmaRange(D);
		SourceRange WholeRange = D->getInnermostCapturedStmt()->getCapturedStmt()->getSourceRange();

		std::string BStr;
		llvm::raw_string_ostream BeginStr(BStr);
		/* 
		 * Function Prototype
		 */

		BeginStr << "\n{\n";
		/* 
		 * Implicit Params
		 */
		for (auto *param : CaptDecl->parameters()) {
			BeginStr << "//";
			param->print(BeginStr, Policy);
			BeginStr << "\n";
		}
		BeginStr << "\n";

		/*
		 * num_teams & thread_limit clauses
		 */
		const OMPNumTeamsClause *NumTeamsClause = D->getSingleClause<OMPNumTeamsClause>();
		const OMPThreadLimitClause *ThreadLimitClause = D->getSingleClause<OMPThreadLimitClause>();

		BeginStr << "\n\n// team configuration\n";
		if (NumTeamsClause) {
			BeginStr << "PushNumTeams(";
			NumTeamsClause->getNumTeams()->printPretty(BeginStr, nullptr, Policy);
			BeginStr << ");\n";
		} else {
			BeginStr << "PushNumTeams();\n";
		}

		if (ThreadLimitClause) {
			BeginStr << "PushThreadLimit(";
			ThreadLimitClause->getThreadLimit()->printPretty(BeginStr, nullptr, Policy);
			BeginStr << ");\n";
		} else {
			BeginStr << "PushThreadLimit();\n";
		}
		//			BeginStr << 
		BeginStr << "\n";

		/*
		 * Kernel call Body
		 */
		InsertTextAfter(PragmaRange.getEnd(), BeginStr.str());

		std::string EStr;
		llvm::raw_string_ostream EndStr(EStr);
		EndStr << "\n\n// team configuration clean up\n";
		if (NumTeamsClause) {
			EndStr << "PopNumTeams(";
			NumTeamsClause->getNumTeams()->printPretty(EndStr, nullptr, Policy);
			EndStr << ");\n";
		} else {
			EndStr << "PopNumTeams();\n";
		}

		if (ThreadLimitClause) {
			EndStr << "PopThreadLimit(";
			ThreadLimitClause->getThreadLimit()->printPretty(EndStr, nullptr, Policy);
			EndStr << ");\n";
		} else {
			EndStr << "PopThreadLimit();\n";
		}


		EndStr << "\n}\n";
		InsertBeforeToken(WholeRange.getEnd(), EndStr.str());
	}

	void TransformExecutableDirectiveOfTarget (OMPExecutableDirective *D) {
		CapturedStmt *CaptStmt = D->getInnermostCapturedStmt();
		CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
		PrintingPolicy Policy(Rewriter_.getLangOpts());
		Stmt *Body = CaptDecl->getBody();
		SourceRange PragmaRange = getPragmaRange(D);
		SourceRange WholeRange = D->getInnermostCapturedStmt()->getCapturedStmt()->getSourceRange();

		std::string BStr;
		llvm::raw_string_ostream BeginStr(BStr);

		/* 
		 * Function Prototype
		 */
		BeginStr << "\n{\n";
		/* 
		 * Implicit Params
		 */
		for (auto *param : CaptDecl->parameters()) {
			BeginStr << "//";
			param->print(BeginStr, Policy);
			BeginStr << "\n";
		}
		BeginStr << "\n";

		/*
		 * Mapping variables
		 */
		BeginStr << "\n// Initial Memory Management\n";
		for (auto *clause: D->getClausesOfKind<OMPMapClause>()) {
			OpenMPMapClauseKind kind = clause->getMapType();
			bool copy = (kind == OMPC_MAP_to) || (kind == OMPC_MAP_tofrom);
			for (auto comp_pair : clause->component_lists()) {
				const ValueDecl* d = comp_pair.first;
				for (auto e : comp_pair.second) {
					std::string created_name_str;
					llvm::raw_string_ostream created_name(created_name_str);
					created_name << "created_" << d->getName();
					Expr *expr = e.getAssociatedExpression();
					BeginStr << "int " << created_name.str() << ";\n";
					BeginStr << d->getType().getAsString() << " m_";
					BeginStr << d->getNameAsString() << " = ";
					BeginStr << "CreateOrGetBuffer(";

					if (OMPArraySectionExpr *section = dyn_cast<OMPArraySectionExpr>(expr)) {
						// FIXME: How about pointers?
						std::string startAddress_str;
						llvm::raw_string_ostream startAddress(startAddress_str);
						section->getBase()->printPretty(startAddress, nullptr, Policy);
						startAddress << "[";
						section->getLowerBound()->printPretty(startAddress, nullptr, Policy);
						startAddress << "]";

						BeginStr << "&";
						BeginStr << startAddress.str();
						BeginStr << ", ";
						section->getLength()->printPretty(BeginStr, nullptr, Policy);
						BeginStr << " * sizeof(" << startAddress.str() << "), ";
						BeginStr << (copy ? "1" : "0");
						BeginStr << ", ";
						BeginStr << "&" << created_name.str();
						BeginStr << ");";
						BeginStr << "\n";
						break;
					} else if (DeclRefExpr *var = dyn_cast<DeclRefExpr>(expr)) {
						BeginStr << "&";
						var->printPretty(BeginStr, nullptr, Policy);
						BeginStr << ", sizeof(";
						var->printPretty(BeginStr, nullptr, Policy);
						BeginStr <<"), " << (copy ? "1" : "0");
						BeginStr <<", " << created_name.str();
						BeginStr << ");";
						BeginStr << "\n";
					} else {
						llvm_unreachable("map clause contains only ArraySection or a value");
					}

				}
			}
		}

		BeginStr << "\n";
		InsertTextBefore(PragmaRange.getEnd(), BeginStr.str());
		/*
		 * Body
		 */
		std::string EStr;
		llvm::raw_string_ostream EndStr(EStr);
		EndStr << "\n// Memory cleanup\n";
		for (auto *clause: D->getClausesOfKind<OMPMapClause>()) {
			OpenMPMapClauseKind kind = clause->getMapType();
			bool copy = (kind == OMPC_MAP_to) || (kind == OMPC_MAP_tofrom);
			for (auto comp_pair : clause->component_lists()) {
				const ValueDecl* d = comp_pair.first;
				for (auto e : comp_pair.second) {
					std::string created_name_str;
					llvm::raw_string_ostream created_name(created_name_str);
					created_name << "created_" << d->getName();
					Expr *expr = e.getAssociatedExpression();
					EndStr << "DestroyBuffer(";

					if (OMPArraySectionExpr *section = dyn_cast<OMPArraySectionExpr>(expr)) {
						std::string startAddress_str;
						llvm::raw_string_ostream startAddress(startAddress_str);
						section->getBase()->printPretty(startAddress, nullptr, Policy);
						startAddress << "[";
						section->getLowerBound()->printPretty(startAddress, nullptr, Policy);
						startAddress << "]";

						EndStr << "&";
						EndStr << ", ";
						section->getLength()->printPretty(EndStr, nullptr, Policy);
						EndStr << " * sizeof(" << startAddress.str() << "), ";
						EndStr << (copy ? "1" : "0");
						EndStr << ", ";
						EndStr << created_name.str();
						EndStr << ");";
						EndStr << "\n";
						break;
					} else if (DeclRefExpr *var = dyn_cast<DeclRefExpr>(expr)) {
						EndStr << "&";
						var->printPretty(EndStr, nullptr, Policy);
						EndStr << ", sizeof(";
						var->printPretty(EndStr, nullptr, Policy);
						EndStr <<"), " << (copy ? "1" : "0");
						EndStr <<", " << created_name.str();
						EndStr << ");";
						EndStr << "\n";
					} else {
						llvm_unreachable("map clause contains only ArraySection or a value");
					}
				}
			}
		}
		EndStr << "\n}\n";
		InsertBeforeToken(WholeRange.getEnd(), EndStr.str());
	}


	void RewriteTargetTeamsDistributeDirective(OMPTargetTeamsDistributeDirective *D) {
		CommentExecutableDirective(D);
		TransformExecutableDirectiveOfTarget(D);
		TransformExecutableDirectiveOfTeams(D);
		TransformExecutableDirectiveOfDistribute(D);
	}

	void Initialize(SourceManager &SrcMgr, LangOptions &LOpts) {
		Rewriter_.setSourceMgr(SrcMgr, LOpts);

		StringRef TempFilePrefix = "__omp_tmp";
		StringRef TempFileSuffix = "h";

		// Make Temp file
		SmallString<32> ResultPath;
		std::error_code err = llvm::sys::fs::createTemporaryFile(TempFilePrefix, TempFileSuffix, ResultPath);
		const FileEntry *FE = SrcMgr.getFileManager().getFile(ResultPath.str(), /*OpenFile=*/false);

		assert(FE && "Cannot Create a file");

		StringRef FileName = FE->getName();

		MainFileID_ = SrcMgr.getMainFileID();
		KernelFileID_ = SrcMgr.getOrCreateFileID(FE, SrcMgr::C_User);

		assert(MainFileID_.isValid() && "Main file is not valid");
		assert(KernelFileID_.isValid() && "Temporary file is not valid");

		// This is necessary to generate nonempty buffer
		Rewriter_.getEditBuffer(MainFileID_);
		Rewriter_.getEditBuffer(KernelFileID_);


		SourceLocation StartLoc = SrcMgr.getLocForStartOfFile(MainFileID_);
		InsertTextAfter(StartLoc, "#include <cuda_runtime.h>\n");
		InsertTextAfter(StartLoc, "#include \"");
		InsertTextAfter(StartLoc, FileName);
		InsertTextAfter(StartLoc, "\"\n");
		//SourceLocation TestLoc = SrcMgr.getLocForStartOfFile(KernelFileID_);
		//InsertTextAfter(TestLoc, "this is a test text\n");
	}

	void Finalize() {
		std::error_code EC;
		llvm::raw_fd_ostream MainFileStream(Rewriter_.getSourceMgr().getFileEntryForID(MainFileID_)->getName(), EC);
		assert(!EC && "cannot open file ostream");
		llvm::raw_fd_ostream KernelFileStream(Rewriter_.getSourceMgr().getFileEntryForID(KernelFileID_)->getName(), EC);
		assert(!EC && "cannot open file ostream");

		const RewriteBuffer *MainBuf = Rewriter_.getRewriteBufferFor(MainFileID_);
		const RewriteBuffer *KernelBuf = Rewriter_.getRewriteBufferFor(KernelFileID_);
		assert(MainBuf && "MainBuf is Null");
		assert(KernelBuf && "KernelBuf is Null");

		MainBuf->write(MainFileStream);
		KernelBuf->write(KernelFileStream);
	}

	private:
	FileID MainFileID_;
	FileID KernelFileID_;
	Rewriter Rewriter_;

};
