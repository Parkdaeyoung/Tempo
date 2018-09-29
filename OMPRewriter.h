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
	/* Public Interfaces */
	void RewriteTargetDataDirective(OMPTargetDataDirective *D) {
		CommentDirective(D);
		TransformTargetDataDirective(D);
	}

	void RewriteTargetDirective(OMPTargetDirective *D) {
		CommentDirective(D);
		TransformTargetDirective(D);
	}

	void RewriteTeamsDirective(OMPTeamsDirective *D) {
		CommentDirective(D);
		TransformTeamsDirective(D);
	}

	void RewriteDistributeDirective(OMPDistributeDirective *D) {
		CommentDirective(D);
		TransformDistributeDirective(D);
	}

	void RewriteTargetTeamsDirective(OMPTargetTeamsDirective *D) {
		CommentDirective(D);
		TransformTargetDirective(D);
		TransformTeamsDirective(D);
	}

	void RewriteTargetTeamsDistributeDirective(OMPTargetTeamsDistributeDirective *D) {
		CommentDirective(D);
		TransformTargetDirective(D);
		TransformTeamsDirective(D);
		TransformDistributeDirective(D);
	}

	void RewriteTeamsDistributeParallelForDirective(OMPTeamsDistributeParallelForDirective *D) {
		CommentDirective(D);
		TransformTeamsDirective(D);
		TransformDistributeParallelForDirective(D);
	}

	void RewriteDistributeParallelForDirective(OMPDistributeParallelForDirective *D) {
		CommentDirective(D);
		TransformDistributeParallelForDirective(D);
	}

	void RewriteTargetTeamsDistributeParallelForDirective(OMPTargetTeamsDistributeParallelForDirective *D) {
		CommentDirective(D);
		TransformTargetDirective(D);
		TransformTeamsDirective(D);
		TransformDistributeParallelForDirective(D);
	}

	void Initialize(SourceManager &SrcMgr, LangOptions &LOpts) {
		StringRef HostFilePrefix = "__o2c_host";
		StringRef HostFileSuffix = "cu";
		StringRef KernelFilePrefix = "__o2c_kernel";
		StringRef KernelFileSuffix = "h";

		// Make Temp file
		std::error_code err;
		SmallString<32> ResultPath;
		//		err = llvm::sys::fs::createTemporaryFile(HostFilePrefix,
		//			   	HostFileSuffix, ResultPath);
		//		const FileEntry *HostFE = SrcMgr.getFileManager().getFile(ResultPath.str(), /*OpenFile=*/false);
		err = llvm::sys::fs::createTemporaryFile(KernelFilePrefix,
				KernelFileSuffix, ResultPath);
		const FileEntry *KernelFE = SrcMgr.getFileManager().getFile(ResultPath.str(), /*OpenFile=*/false);

		//		assert(HostFE && "Cannot Create a file");
		assert(KernelFE && "Cannot Create a file");


		// Copy input File To HostFile
		//		StringRef InputFileName = SrcMgr.getFileEntryForID(SrcMgr.getMainFileID())->getName();
		//		llvm::sys::fs::copy_file(InputFileName, HostFE->getName());


		//		HostFileID_ = SrcMgr.getOrCreateFileID(HostFE, SrcMgr::C_User);
		HostFileID_ = SrcMgr.getMainFileID();
		KernelFileID_ = SrcMgr.getOrCreateFileID(KernelFE, SrcMgr::C_User);
		//		SrcMgr.setMainFileID(HostFileID_);

		assert(HostFileID_.isValid() && "Main file is not valid");
		assert(KernelFileID_.isValid() && "Temporary file is not valid");

		Rewriter_.setSourceMgr(SrcMgr, LOpts);

		StringRef KernelFileName = KernelFE->getName();
		SourceLocation StartLoc = SrcMgr.getLocForStartOfFile(HostFileID_);
		InsertTextAfter(StartLoc, "#include \"");
		InsertTextAfter(StartLoc, KernelFileName);
		InsertTextAfter(StartLoc, "\"\n");

		SourceLocation TestLoc = SrcMgr.getLocForStartOfFile(KernelFileID_);
		InsertTextAfter(TestLoc, "#include <cuda_runtime.h>\n");
	}

	void Finalize() {
		//		std::error_code EC;
		//		llvm::raw_fd_ostream MainFileStream(Rewriter_.getSourceMgr().getFileEntryForID(HostFileID_)->getName(), EC);
		//		assert(!EC && "cannot open file ostream");
		//		llvm::raw_fd_ostream KernelFileStream(Rewriter_.getSourceMgr().getFileEntryForID(KernelFileID_)->getName(), EC);
		//		assert(!EC && "cannot open file ostream");

		const RewriteBuffer *HostBuf = Rewriter_.getRewriteBufferFor(HostFileID_);
		const RewriteBuffer *KernelBuf = Rewriter_.getRewriteBufferFor(KernelFileID_);
		assert(HostBuf && "HostBuf is Null");
		assert(KernelBuf && "KernelBuf is Null");

		//		HostBuf->write(MainFileStream);
		//		KernelBuf->write(KernelFileStream);

		HostBuf->write(llvm::outs());
		KernelBuf->write(llvm::outs());
	}

	private:

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

	void CommentDirective(OMPExecutableDirective *D) {
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


	void TransformDistributeParallelForDirective(OMPExecutableDirective *D) {
		assert(isa<OMPLoopDirective>(D) && "TransformDistributeDirective Failed");

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
		SourceRange CandidateRange = SourceRange(Body->getBeginLoc(), Body->getEndLoc());
		RemoveText(CandidateRange, removeOpts);



		//StringRef Kernel = Rewriter_.getRangeSize(CandidateRange, removeOpts);
		SourceLocation KernelStartLoc = Rewriter_.getSourceMgr().getLocForStartOfFile(KernelFileID_);
		// FIXME

		// Insert Kernel Call;
		std::string CallStr;
		llvm::raw_string_ostream CallSStream(CallStr);
		CallSStream << "{\n";
		int pos = 0;
		for (auto capt : CaptStmt->capture_inits()) {
			CallSStream << "SetArgument(&";
			capt->printPretty(CallSStream, nullptr, Policy);
			CallSStream << ", " << pos;
			CallSStream << ");\n";
			pos++;
		}
		CallSStream << "LaunchKernel();\n";
		CallSStream << "}\n";

		InsertTextAfter(Body->getBeginLoc(), CallSStream.str());




		// Insert Kernel function
		std::string KernelStr;
		llvm::raw_string_ostream KernelSStream(KernelStr);
		KernelSStream << "__global__ void KernelName(";

		unsigned count = 0;
		for (auto capt : CaptStmt->captures()) {
			VarDecl *decl = capt.getCapturedVar();
			KernelSStream << decl->getType().getAsString();
			KernelSStream << " ";
			KernelSStream << decl->getName();
			if (count < CaptStmt->capture_size()-1)
				KernelSStream << ", ";
			count ++;
		}
		KernelSStream << ")\n";
		KernelSStream << "{\n";
		KernelSStream << "/*\n";
		LoopDirective->printPretty(KernelSStream, nullptr, Policy);
		KernelSStream << "\n*/\n";


		KernelSStream << "int __o2c_gid = blockDim.x * blockIdx.x + threadIdx.x;\n";
		KernelSStream << "int __o2c_gsize = blockDim.x * gridDim.x;\n";

		//FIXME: Type Infomation Must be Added!!!
		for (auto e : LoopDirective->inits()) {
			e->printPretty(KernelSStream, nullptr, Policy);
			KernelSStream << ";\n";
		}
		KernelSStream << "for (";
		KernelSStream << "int __o2c_i = __o2c_gid; \n";
		KernelSStream << "__o2c_i <= ";
		LoopDirective->getLastIteration()->printPretty(KernelSStream, nullptr, Policy);
		KernelSStream << "; \n";
		KernelSStream << "__o2c_i += __o2c_gsize) {\n";
		for (auto e : LoopDirective->updates()) {
			e->printPretty(KernelSStream, nullptr, Policy);
			KernelSStream << ";\n";
		}
		LoopDirective->getBody()->printPretty(KernelSStream, nullptr, Policy);
		KernelSStream << "\n}\n"; // End of Forloop

		KernelSStream << "\n}\n"; // End of Kernel
		InsertTextAfter(KernelStartLoc, KernelSStream.str());
	}

	void TransformDistributeDirective(OMPExecutableDirective *D) {
		assert(isa<OMPLoopDirective>(D) && "TransformDistributeDirective Failed");

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
		SourceRange CandidateRange = SourceRange(Body->getBeginLoc(), Body->getEndLoc());
		RemoveText(CandidateRange, removeOpts);



		//StringRef Kernel = Rewriter_.getRangeSize(CandidateRange, removeOpts);
		SourceLocation KernelStartLoc = Rewriter_.getSourceMgr().getLocForStartOfFile(KernelFileID_);
		// FIXME

		// Insert Kernel Call;
		std::string CallStr;
		llvm::raw_string_ostream CallSStream(CallStr);
		CallSStream << "{\n";
		for (auto capt : CaptStmt->capture_inits()) {
			//			VarDecl *decl = capt.getCapturedVar();
			CallSStream << "SetArgument(&";
			capt->printPretty(CallSStream, nullptr, Policy);
			//			decl->print(CallSStream);
			CallSStream << ");\n";
		}
		CallSStream << "LaunchKernel();\n";
		CallSStream << "}\n";

		InsertTextAfter(Body->getBeginLoc(), CallSStream.str());




		// Insert Kernel function
		std::string KernelStr;
		llvm::raw_string_ostream KernelSStream(KernelStr);
		KernelSStream << "__global__ void KernelName(";

		unsigned count = 0;
		for (auto capt : CaptStmt->captures()) {
			VarDecl *decl = capt.getCapturedVar();
			KernelSStream << decl->getType().getAsString();
			KernelSStream << " ";
			KernelSStream << decl->getName();
			if (count < CaptStmt->capture_size()-1)
				KernelSStream << ", ";
			count ++;
		}
		KernelSStream << ")\n";
		KernelSStream << "{\n";
		KernelSStream << "/*\n";
		LoopDirective->printPretty(KernelSStream, nullptr, Policy);
		KernelSStream << "\n*/\n";


		KernelSStream << "int __o2c_lid = blockDim.x * blockIdx.x + threadIdx.x;\n";
		KernelSStream << "int __o2c_lsize = blockDim.x;\n";
		KernelSStream << "for (";
		KernelSStream << "int __o2c_i = __o2c_lid; \n";
		KernelSStream << "__o2c_i <= ";
		LoopDirective->getLastIteration()->printPretty(KernelSStream, nullptr, Policy);
		KernelSStream << "; \n";
		KernelSStream << "__o2c_i += __o2c_lsize) {\n";
		for (auto e : LoopDirective->updates()) {
			e->printPretty(KernelSStream, nullptr, Policy); // TODO: Macro Reduction?
			KernelSStream << ";\n";
		}
		LoopDirective->getBody()->printPretty(KernelSStream, nullptr, Policy);
		KernelSStream << "\n}\n"; // End of ForLoop
		KernelSStream << "\n}\n"; // End of Kernel
		InsertTextAfter(KernelStartLoc, KernelSStream.str());
	}

	void TransformTeamsDirective (OMPExecutableDirective *D) {
		CapturedStmt *CaptStmt = D->getInnermostCapturedStmt();
		CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
		PrintingPolicy Policy(Rewriter_.getLangOpts());
		Stmt *Body = CaptDecl->getBody();
		SourceRange PragmaRange = getPragmaRange(D);
		SourceRange WholeRange = D->getInnermostCapturedStmt()->getCapturedStmt()->getSourceRange();

		std::string BStr;
		llvm::raw_string_ostream BeginStr(BStr);

		BeginStr << "\n{\n";

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

	void TransformTargetDataDirective (OMPTargetDataDirective *D) {
		CapturedStmt *CaptStmt = D->getInnermostCapturedStmt();
		CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
		PrintingPolicy Policy(Rewriter_.getLangOpts());
		Stmt *Body = CaptDecl->getBody();
		SourceRange PragmaRange = getPragmaRange(D);
		SourceRange WholeRange = D->getInnermostCapturedStmt()->getCapturedStmt()->getSourceRange();


		/*
		 * Makeup environment
		 */
		std::string MakeupStr;
		llvm::raw_string_ostream MakeupSStream(MakeupStr);

		MakeupSStream << "\n{\n";
		MakeupSStream << "//  Make up data environment\n";

		/*
		 * Makeup Environment
		 */
		for (auto *clause: D->getClausesOfKind<OMPMapClause>()) {
			OpenMPMapClauseKind kind = clause->getMapType();
			bool copy = (kind == OMPC_MAP_to) || (kind == OMPC_MAP_tofrom);
			for (auto comp_pair : clause->component_lists()) {
				const ValueDecl* d = comp_pair.first;
				for (auto e : comp_pair.second) {
					std::string CreatedNameStr;
					std::string BufferNameStr;
					llvm::raw_string_ostream CreatedNameSStream(CreatedNameStr);
					llvm::raw_string_ostream BufferNameSStream(BufferNameStr);

					CreatedNameSStream << "created_" << d->getName();
					BufferNameSStream << "__o2c_device_" << d->getNameAsString();

					MakeupSStream << "int" << " " << CreatedNameSStream.str() << ";\n";
					MakeupSStream << d->getType().getAsString() << " " << BufferNameSStream.str();
					MakeupSStream << " = ";
					MakeupSStream << "CreateOrGetBuffer(";

					Expr *expr = e.getAssociatedExpression();
					if (OMPArraySectionExpr *section = dyn_cast<OMPArraySectionExpr>(expr)) {
						std::string AddressStr;
						llvm::raw_string_ostream AddressSStream(AddressStr);
						section->getBase()->printPretty(AddressSStream, nullptr, Policy);
						AddressSStream << "[";
						section->getLowerBound()->printPretty(AddressSStream, nullptr, Policy);
						AddressSStream << "]";

						MakeupSStream << "&";
						MakeupSStream << AddressSStream.str();
						MakeupSStream << ", ";
						section->getLength()->printPretty(MakeupSStream, nullptr, Policy);
						MakeupSStream << " * sizeof(" << AddressSStream.str() << "), ";
						MakeupSStream << (copy ? "1" : "0");
						MakeupSStream << ", ";
						MakeupSStream << "&" << CreatedNameSStream.str();
						MakeupSStream << ");";
						MakeupSStream << "\n";
						break;
					} else if (DeclRefExpr *var = dyn_cast<DeclRefExpr>(expr)) {
						MakeupSStream << "&";
						var->printPretty(MakeupSStream, nullptr, Policy);
						MakeupSStream << ", sizeof(";
						var->printPretty(MakeupSStream, nullptr, Policy);
						MakeupSStream <<"), " << (copy ? "1" : "0");
						MakeupSStream <<", " << CreatedNameSStream.str();
						MakeupSStream << ");";
						MakeupSStream << "\n";
					} else {
						llvm_unreachable("map clause contains only ArraySection or a scalar value");
					}

				}
			}
		}

		MakeupSStream << "\n";
		InsertTextBefore(PragmaRange.getEnd(), MakeupSStream.str());

		/*
		 * Cleanup Environment
		 */
		std::string CleanupStr;
		llvm::raw_string_ostream CleanupSStream(CleanupStr);
		CleanupSStream << "\n// Memory cleanup\n";
		for (auto *clause: D->getClausesOfKind<OMPMapClause>()) {
			OpenMPMapClauseKind kind = clause->getMapType();
			bool copy = (kind == OMPC_MAP_to) || (kind == OMPC_MAP_tofrom);
			for (auto comp_pair : clause->component_lists()) {
				const ValueDecl* d = comp_pair.first;
				for (auto e : comp_pair.second) {
					std::string CreatedNameStr;
					std::string BufferNameStr;
					llvm::raw_string_ostream CreatedNameSStream(CreatedNameStr);
					llvm::raw_string_ostream BufferNameSStream(BufferNameStr);

					CreatedNameSStream << "created_" << d->getName();
					BufferNameSStream << "__o2c_device_" << d->getNameAsString();

					CleanupSStream << "DestroyBuffer(";
					Expr *expr = e.getAssociatedExpression();
					if (OMPArraySectionExpr *section = dyn_cast<OMPArraySectionExpr>(expr)) {
						std::string AddressStr;
						llvm::raw_string_ostream AddressSStream(AddressStr);
						section->getBase()->printPretty(AddressSStream, nullptr, Policy);
						AddressSStream << "[";
						section->getLowerBound()->printPretty(AddressSStream, nullptr, Policy);
						AddressSStream << "]";

						CleanupSStream << "&";
						CleanupSStream << BufferNameSStream.str();
						CleanupSStream << ", ";
						section->getLength()->printPretty(CleanupSStream, nullptr, Policy);
						CleanupSStream << " * sizeof(" << AddressSStream.str() << "), ";
						CleanupSStream << (copy ? "1" : "0");
						CleanupSStream << ", ";
						CleanupSStream << CreatedNameSStream.str();
						CleanupSStream << ");";
						CleanupSStream << "\n";
					} else if (DeclRefExpr *var = dyn_cast<DeclRefExpr>(expr)) {
						CleanupSStream << "&";
						var->printPretty(CleanupSStream, nullptr, Policy);
						CleanupSStream << ", sizeof(";
						var->printPretty(CleanupSStream, nullptr, Policy);
						CleanupSStream <<"), " << (copy ? "1" : "0");
						CleanupSStream <<", " << CreatedNameSStream.str();
						CleanupSStream << ");";
						CleanupSStream << "\n";
					} else {
						llvm_unreachable("map clause contains only ArraySection or a value");
					}
				}
			}
		}
		CleanupSStream << "\n}\n";
		InsertBeforeToken(WholeRange.getEnd(), CleanupSStream.str());
	}

	void TransformTargetDirective (OMPExecutableDirective *D) {
		CapturedStmt *CaptStmt = D->getInnermostCapturedStmt();
		CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
		PrintingPolicy Policy(Rewriter_.getLangOpts());
		Stmt *Body = CaptDecl->getBody();
		SourceRange PragmaRange = getPragmaRange(D);
		SourceRange WholeRange = D->getInnermostCapturedStmt()->getCapturedStmt()->getSourceRange();


		/*
		 * Makeup environment
		 */
		std::string MakeupStr;
		llvm::raw_string_ostream MakeupSStream(MakeupStr);

		MakeupSStream << "\n{\n";
		MakeupSStream << "//  Make up data environment\n";

		// If a defaultmap(tofrom:scalar) is not present,
		// scalar is not mapped, but just firstprivated
		// Otherwise, a scalar variable is treated
		// as if map(tofrom: ...)
		// Any non-scalar variable is treated as if map(tofrom: ....)
		if (D->getSingleClause<OMPDefaultmapClause>()) {
			for (auto capt : CaptStmt->captures()) {
				VarDecl *decl = capt.getCapturedVar();
				if (decl->getType().getTypePtr()->isScalarType()) {
					decl->print(MakeupSStream);
					MakeupSStream <<";\n";
				}
			}
		}
		MakeupSStream << "\n// Initial Memory Management\n";
		for (auto *clause: D->getClausesOfKind<OMPMapClause>()) {
			OpenMPMapClauseKind kind = clause->getMapType();
			bool copy = (kind == OMPC_MAP_to) || (kind == OMPC_MAP_tofrom);
			for (auto comp_pair : clause->component_lists()) {
				const ValueDecl* d = comp_pair.first;
				for (auto e : comp_pair.second) {
					std::string CreatedNameStr;
					std::string BufferNameStr;
					llvm::raw_string_ostream CreatedNameSStream(CreatedNameStr);
					llvm::raw_string_ostream BufferNameSStream(BufferNameStr);

					CreatedNameSStream << "created_" << d->getName();
					BufferNameSStream << "__o2c_device_" << d->getNameAsString();

					MakeupSStream << "int" << " " << CreatedNameSStream.str() << ";\n";
					MakeupSStream << d->getType().getAsString() << " " << BufferNameSStream.str();
					MakeupSStream << " = ";
					MakeupSStream << "CreateOrGetBuffer(";

					Expr *expr = e.getAssociatedExpression();
					if (OMPArraySectionExpr *section = dyn_cast<OMPArraySectionExpr>(expr)) {
						// FIXME: How about pointers?
						std::string AddressStr;
						llvm::raw_string_ostream AddressSStream(AddressStr);
						section->getBase()->printPretty(AddressSStream, nullptr, Policy);
						AddressSStream << "[";
						section->getLowerBound()->printPretty(AddressSStream, nullptr, Policy);
						AddressSStream << "]";

						MakeupSStream << "&";
						MakeupSStream << AddressSStream.str();
						MakeupSStream << ", ";
						section->getLength()->printPretty(MakeupSStream, nullptr, Policy);
						MakeupSStream << " * sizeof(" << AddressSStream.str() << "), ";
						MakeupSStream << (copy ? "1" : "0");
						MakeupSStream << ", ";
						MakeupSStream << "&" << CreatedNameSStream.str();
						MakeupSStream << ");";
						MakeupSStream << "\n";
						break;
					} else if (DeclRefExpr *var = dyn_cast<DeclRefExpr>(expr)) {
						MakeupSStream << "&";
						var->printPretty(MakeupSStream, nullptr, Policy);
						MakeupSStream << ", sizeof(";
						var->printPretty(MakeupSStream, nullptr, Policy);
						MakeupSStream <<"), " << (copy ? "1" : "0");
						MakeupSStream <<", " << CreatedNameSStream.str();
						MakeupSStream << ");";
						MakeupSStream << "\n";
					} else {
						llvm_unreachable("map clause contains only ArraySection or a value");
					}

				}
			}
		}

		MakeupSStream << "\n";
		InsertTextBefore(PragmaRange.getEnd(), MakeupSStream.str());



		/*
 		 * Cleanup environment
		 */

		std::string CleanupStr;
		llvm::raw_string_ostream CleanupSStream(CleanupStr);
		CleanupSStream << "\n// Memory cleanup\n";
		for (auto *clause: D->getClausesOfKind<OMPMapClause>()) {
			OpenMPMapClauseKind kind = clause->getMapType();
			bool copy = (kind == OMPC_MAP_to) || (kind == OMPC_MAP_tofrom);
			for (auto comp_pair : clause->component_lists()) {
				const ValueDecl* d = comp_pair.first;
				for (auto e : comp_pair.second) {
					std::string CreatedNameStr;
					std::string BufferNameStr;
					llvm::raw_string_ostream CreatedNameSStream(CreatedNameStr);
					llvm::raw_string_ostream BufferNameSStream(BufferNameStr);

					CreatedNameSStream << "created_" << d->getName();
					BufferNameSStream << "__o2c_device_" << d->getNameAsString();

					CleanupSStream << "DestroyBuffer(";
					Expr *expr = e.getAssociatedExpression();
					if (OMPArraySectionExpr *section = dyn_cast<OMPArraySectionExpr>(expr)) {
						std::string AddressStr;
						llvm::raw_string_ostream AddressSStream(AddressStr);
						section->getBase()->printPretty(AddressSStream, nullptr, Policy);
						AddressSStream << "[";
						section->getLowerBound()->printPretty(AddressSStream, nullptr, Policy);
						AddressSStream << "]";

						CleanupSStream << "&";
						CleanupSStream << BufferNameSStream.str();
						CleanupSStream << ", ";
						section->getLength()->printPretty(CleanupSStream, nullptr, Policy);
						CleanupSStream << " * sizeof(" << AddressSStream.str() << "), ";
						CleanupSStream << (copy ? "1" : "0");
						CleanupSStream << ", ";
						CleanupSStream << CreatedNameSStream.str();
						CleanupSStream << ");";
						CleanupSStream << "\n";
						break;
					} else if (DeclRefExpr *var = dyn_cast<DeclRefExpr>(expr)) {
						CleanupSStream << "&";
						var->printPretty(CleanupSStream, nullptr, Policy);
						CleanupSStream << ", sizeof(";
						var->printPretty(CleanupSStream, nullptr, Policy);
						CleanupSStream <<"), " << (copy ? "1" : "0");
						CleanupSStream <<", " << CreatedNameSStream.str();
						CleanupSStream << ");";
						CleanupSStream << "\n";
					} else {
						llvm_unreachable("map clause contains only ArraySection or a value");
					}
				}
			}
		}
		CleanupSStream << "\n}\n";
		InsertBeforeToken(WholeRange.getEnd(), CleanupSStream.str());
	}

	FileID HostFileID_;
	FileID KernelFileID_;
	Rewriter Rewriter_;

};
