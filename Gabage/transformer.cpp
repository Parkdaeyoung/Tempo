//-------------------------------------------------------------------------
//
// rewritersample.cpp: Source-to-source transformation sample with Clang,
// using Rewriter - the code rewriting interface.
//
// Eli Bendersky (eliben@gmail.com)
// This code is in the public domain
//
#include <cstdio>
#include <string>
#include <sstream>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace std;

class TransformVisitor : public RecursiveASTVisitor<TransformVisitor>
{
	public:
		TransformVisitor(Rewriter &_Rewriter) : _Rewriter(_Rewriter) {}

		bool VisitOMPTargetDataDirective(OMPTargetDataDirective *D) {
			/*
			 * #pragma omp target data clause[[[,] clause] ... ] new-line
			 * 	strucutured-block
			 * where clause is one of following:
			 * 	if([target data:] scalar-expression)
			 * 	device(integer-expression)
			 * 	map([[map-type-modifier[,]] map-type:] list)
			 * 	use_device_ptr(list)
			 */

			llvm::outs() << "TargetDataDirective\n";
			for (auto clause : D->clauses()) {
				if (OMPMapClause::classof(clause)) { // Map clause
					OMPMapClause *MapClause = cast<OMPMapClause>(clause);

					llvm::outs() 
						<< getOpenMPSimpleClauseTypeName(clause->getClauseKind(),MapClause->getMapType())
						<< ": ";

					for (auto e : MapClause->children()) {
						OMPArraySectionExpr *ArrSection = cast<OMPArraySectionExpr>(e);
						llvm::outs() << "base=" << ArrSection->getBase()->getStmtClassName();
						llvm::outs() << ", lower bound= " << ArrSection->getLowerBound()->getStmtClassName();
						llvm::outs() << ", length= " << ArrSection->getLength()->getStmtClassName();
					}
					llvm::outs() << "\n";
				}
			}
			return true;
		}

		bool VisitOMPTargetDirective(OMPTargetDirective *D) {
			llvm::outs() << "VisitOMPTargetDirective\n";
			return true;
		}

		bool VisitOMPTargetEnterDataDirective(OMPTargetEnterDataDirective *D) {
			llvm::outs() << "VisitOMPTargetEnterDataDirective\n";
			return true;
		}

		bool VisitOMPTargetExitDataDirective(OMPTargetExitDataDirective *D) {
			llvm::outs() << "VisitOMPTargetExitDataDirective\n";
			return true;
		}

		bool VisitOMPTargetParallelForDirective(OMPTargetParallelForDirective *D) {
			llvm::outs() << "VisitOMPTargetParallelForDirective\n";
			return true;
		};

	private:
		Rewriter &_Rewriter;
};

class TransformConsumer : public ASTConsumer
{
	public:
		TransformConsumer(Rewriter &_Rewriter) : Visitor(_Rewriter) {}

		virtual bool HandleTopLevelDecl(DeclGroupRef DR) {
			for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
				// Traverse the declaration using our AST visitor.
				Visitor.TraverseDecl(*b);
			}
			return true;
		}

	private:
		TransformVisitor Visitor;
};

class TransformAction : public ASTFrontendAction {
	public:
		virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
				CompilerInstance &CI, StringRef InFile) {
			CI.createDiagnostics();

			// Initialize target info with the default triple for our platform.
			std::shared_ptr<TargetOptions> TO(new TargetOptions);
			TO->Triple = llvm::sys::getDefaultTargetTriple();
			TargetInfo *TI = TargetInfo::CreateTargetInfo(CI.getDiagnostics(), TO);
			CI.setTarget(TI);

			CI.createFileManager();
			FileManager &FileMgr = CI.getFileManager();
			CI.createSourceManager(FileMgr);
			SourceManager &SourceMgr = CI.getSourceManager();
			CI.createPreprocessor(TU_Complete);
			CI.createASTContext();

			// A Rewriter helps us manage the code rewriting task.
			_Rewriter.setSourceMgr(SourceMgr, CI.getLangOpts());

			// Set the main file handled by the source manager to the input file.
			const FileEntry *FileIn = FileMgr.getFile(InFile);
			FileID FID = SourceMgr.getOrCreateFileID(FileIn, SrcMgr::C_User);
			SourceMgr.setMainFileID(FID);
			CI.getDiagnosticClient().BeginSourceFile(
					CI.getLangOpts(),
					&CI.getPreprocessor());

			return std::unique_ptr<clang::ASTConsumer>(new TransformConsumer(_Rewriter));
		}

		virtual void EndSourceFileAction() {
			const RewriteBuffer *RewriteBuf = _Rewriter.getRewriteBufferFor(
					getCompilerInstance().getSourceManager().getMainFileID());
			llvm::outs() << string(RewriteBuf->begin(), RewriteBuf->end());
		}

	private:
		Rewriter _Rewriter;
};


int main(int argc, char *argv[])
{
	if (argc == 1) {
		llvm::errs() << "Usage: " << argv[1] << " <filename>\n";
		return 1;
	}

	std::vector<std::string> Args(&argv[2], &argv[argc]);
	tooling::runToolOnCodeWithArgs(new TransformAction, argv[1], Args);


	return 0;
}
