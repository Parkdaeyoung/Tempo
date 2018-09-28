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


using namespace clang;
using namespace std;

class OMPTransform : public TreeTransform<OMPTransform> 
{
};


class OMPTransformAction : public ASTFrontendAction {
};

int main(int argc, char **argv)
{
	// FIXME: Stack overflow?
	ArrayRef<const char *>Args(&argv[0], &argv[argc]);
	CompilerInstance CI;
	// 0. Initialization


	// 1. Create Invocation and set
	std::shared_ptr<CompilerInvocation>Inv = createInvocationFromCommandLine(Args);
	CI.setInvocation(Inv);

	// 2. Create DiagnosticsEngine
	CI.createDiagnostics();
	IntrusiveRefCntPtr<DiagnosticsEngine>(&CI.getDiagnostics());


	// 3. Create FileManager & SourceManager
	CI.createFileManager();
	CI.createSourceManager(CI.getFileManager());

	// 4. Execute Action
	OMPTransform TT;
	CI.ExecuteAction(TT);
	return 0;
}

