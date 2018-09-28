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

class OMPVisitor : public RecursiveASTVisitor<OMPVisitor>
{
	public:
		OMPVisitor(CompilerInstance &CI, Rewriter &Rewriter) : _CI(CI), _Rewriter(Rewriter) {}

		void printCaptStmtAsFunction(CapturedStmt *CaptStmt) {
			PrintingPolicy Policy(_CI.getLangOpts());
			CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
			Stmt *Body = CaptDecl->getBody();

			std::string Str;
			llvm::raw_string_ostream ss(Str);
			/* 
			 * Function Prototype
			 */
			ss << "void function(";

			auto it = CaptStmt->capture_begin();
			for ( ; it < CaptStmt->capture_end() - 1
					; it++) {
				it->getCapturedVar()->print(ss, Policy);
				ss << ", ";
			}
			if (it != CaptStmt->capture_end()) {
				it->getCapturedVar()->print(ss, Policy);
			}

			ss << ")\n";

			ss << "{\n";
			/* 
			 * Implicit Params
			 */
			for (auto *param : CaptDecl->parameters()) {
				ss << "//";
				param->print(ss, Policy);
				ss << "\n";
			}
			ss<<"\n";

			/*
			 * Body
			 */
			Body->printPretty(ss, nullptr, Policy, 1);
			ss << "\n}\n";

			// Print out to the stdout
//			llvm::outs() << ss.str() << "\n";
		}

		//		FunctionDecl* CaptStmtToFunctionDecl(CapturedStmt* CaptStmt) {
		//			ASTContext &Ctx = _CI.getASTContext();
		//			PrintingPolicy Policy(_CI.getLangOpts());
		//			CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
		//			Stmt *Body = CaptDecl->getBody();
		//
		//
		//			SmallVector<QualType, 16> ParamTypes;
		//			for (auto capture : CaptStmt->captures()) {
		//				ParamTypes.push_back(capture.getCapturedVar()->getType());
		//			}
		//
		//			FunctionProtoType::ExtProtoInfo EPI;
		//			QualType Ty = Ctx.getFunctionType(Ctx.VoidTy, ParamTypes, EPI);
		//			FunctionDecl* FuncDecl = FunctionDecl::Create(Ctx,
		//				   	Ctx.getTranslationUnitDecl(),
		//				   	SourceLocation(),
		//				   	SourceLocation(),
		//					Ctx.getMakeIntegerSeqName(),
		//					Ty,
		//				   	Ctx.getTrivialTypeSourceInfo(Ty),
		//				   	SC_Static);
		//
		//			int i;
		//			SmallVector<ParmVarDecl*, 16> Params;
		//			for (auto capture : CaptStmt->captures()) {
		//				VarDecl *Var = capture.getCapturedVar();
		//				ParmVarDecl *Param = ParmVarDecl::Create(Ctx,
		//						FuncDecl,
		//					   	SourceLocation(),
		//					   	SourceLocation(),
		//						Var->getIdentifier(),
		//						Var->getType(),
		//						Ctx.getTrivialTypeSourceInfo(Var->getType()),
		//						SC_None,
		//					   	nullptr);
		//
		//
		//				Param->setScopeInfo(0, i);
		//				Params.push_back(Param);
		//				i++;
		//			}
		//			FuncDecl->setParams(Params);
		//			FuncDecl->setBody(Body);
		//			FuncDecl->print(llvm::outs());
		//
		//			return FuncDecl;
		//		}

		//		CallExpr* CallExprToFunctionDecl(FunctionDecl *Decl) {
		//			ASTContext &Ctx = _CI.getASTContext();
		//			PrintingPolicy Policy(_CI.getLangOpts());
		//
		//			bool RefersToEnclosingVariableorCapture = true;
		//			SmallVector<Expr*, 16> Args;
		//			for (auto* param : Decl->parameters()) {
		//				QualType Type = param->getType().getNonReferenceType();
		//				DeclRefExpr *DeclRef = DeclRefExpr::Create(Ctx,
		//						NestedNameSpecifierLoc(),
		//						SourceLocation(),
		//						param,
		//						RefersToEnclosingVariableorCapture,
		//						SourceLocation(),
		//						Type,
		//						VK_LValue
		//						);
		//				Args.push_back(DeclRef);
		//			}
		//
		//			CallExpr *Call = new (Ctx) CallExpr(Ctx,
		//					Decl,
		//					Args,
		//					Decl->getType(),
		//					Expr::getValueKindForType(Decl->getType()),
		//					SourceLocation());
		//			Call->printPretty(llvm::outs(), nullptr, Policy);
		//
		//
		//			return Call;
		//		}


		/*
		 * See OpenMP 4.5 Specification for more information
		 */

		//-------------------------------------------------------------------------//
		// Parallel Construct
		// 		'parallel' Construct
		//-------------------------------------------------------------------------//
		bool VisitOMPParallelDirective(OMPParallelDirective *D) {
			/*
			 * #pragma omp parallel [clause [[,] clause] ...] new-line
			 * 	structured-block
			 *
			 * clause:
			 *
			 * 		if([parallel :] scalar-expression)
			 * 		num_threads(integer-expression)
			 * 		default(shared | none)
			 * 		private(list)
			 * 		firstprivate(list)
			 * 		shared(list)
			 * 		copyin(list)
			 * 		reduction(reduction-identifier : list)
			 * 		proc_bind(master | close | spread)
			 *
			 */
			llvm::outs() << "VisitOMPParallelDirective\n";
			PrintingPolicy Policy(_CI.getLangOpts());
			SourceManager &SrcMgr = _CI.getSourceManager();
			FileID FID = SrcMgr.getFileID(D->getBeginLoc());
			unsigned LineNo = SrcMgr.getPresumedLineNumber(D->getBeginLoc());
			SourceLocation BeginLoc = SrcMgr.translateLineCol(FID, LineNo, 1);
			SourceLocation EndLoc = D->getAssociatedStmt()->getEndLoc();

			std::string Str;
			llvm::raw_string_ostream ss(Str);

			D->getInnermostCapturedStmt()->printPretty(ss, nullptr, Policy, 1);
			_Rewriter.ReplaceText(SourceRange(BeginLoc, EndLoc), "{ abcd; }");

			return true;
		}

		//-------------------------------------------------------------------------//
		// Worksharing Constructs
		// 		'for' Construct
		// 		'sections' Construct
		// 		'section' Construct
		// 		'single' Construct
		//-------------------------------------------------------------------------//
		bool VisitOMPForDirective(OMPForDirective *D) {
			/*
			 * #pragma omp for [clause[[,] clause] ... ] new-line
			 * 	for-loops
			 *
			 * clause:
			 * 		private(list)
			 * 		firstprivate(list)
			 * 		lastprivate(list)
			 * 		linear(list[ :linear-step])
			 * 		reduction(reduction-identifier : list)
			 * 		schedule([modifier [, modifier:]kind[ , chunk_size])
			 * 		collapse(n)
			 * 		ordered[(n)]
			 * 		nowait
			 */
			llvm::outs() << "VisitOMPForDirective\n";

			PrintingPolicy Policy(_CI.getLangOpts());
			SourceManager &SrcMgr = _CI.getSourceManager();
			FileID FID = SrcMgr.getFileID(D->getBeginLoc());
			unsigned LineNo = SrcMgr.getPresumedLineNumber(D->getBeginLoc());
			SourceLocation BeginLoc = SrcMgr.translateLineCol(FID, LineNo, 1);
			SourceLocation EndLoc = D->getAssociatedStmt()->getEndLoc();

			std::string Str;
			llvm::raw_string_ostream ss(Str);

			D->getInnermostCapturedStmt()->printPretty(ss, nullptr, Policy, 1);
			_Rewriter.ReplaceText(SourceRange(BeginLoc, EndLoc), "{ abcd; }");
			return true;
		}
		bool VisitOMPSectionsDirective(OMPSectionsDirective *D) {
			//			llvm::outs() << "VisitOMPSectionsDirective\n";
			return true;
		}

		bool VisitOMPSectionDirective(OMPSectionDirective *D) {
			//			llvm::outs() << "VisitOMPSectionDirective\n";
			return true;
		}

		bool VisitOMPSingleDirective(OMPSingleDirective *D) {
			//			llvm::outs() << "VisitOMPSingleDirective\n";
			return true;
		}

		//-------------------------------------------------------------------------//
		// SIMD Constructs
		// 		'simd' Construct
		// 		'declare simd' Construct
		// 		'for simd' Construct
		//-------------------------------------------------------------------------//
		bool VisitOMPSimdDirective(OMPSimdDirective *D) {
			//			llvm::outs() << "VisitOMPSimdDirective\n";
			return true;
		}

		bool VisitOMPDeclareSimdDeclAttr(OMPDeclareSimdDeclAttr *D) {
			//			llvm::outs() << "VisitOMPDeclareSimdDeclAttr\n";
			return true;
		}

		bool VisitOMPParallelForSimdDirective(OMPParallelForSimdDirective *D) {
			//			llvm::outs() << "VisitOMPParallelForSimdDirective\n";
			return true;
		}

		//-------------------------------------------------------------------------//
		// Tasking Constructs
		// 		'task' Construct
		// 		'taskloop' Construct
		// 		'taskloop simd' Construct
		// 		'taskyield' Construct
		//-------------------------------------------------------------------------//
		bool VisitOMPTaskDirective(OMPTaskDirective *D) {
			//			llvm::outs() << "VisitOMPTaskDirective\n";
			return true;
		}

		bool VisitOMPTaskLoopDirective(OMPTaskLoopDirective *D) {
			//			llvm::outs() << "VisitOMPTaskLoopDirective\n";
			return true;
		}

		bool VisitOMPTaskLoopSimdDirective(OMPTaskLoopSimdDirective *D) {
			//			llvm::outs() << "VisitOMPTaskLoopSimdDirective\n";
			return true;
		}

		bool VisitOMPTaskyieldDirective(OMPTaskyieldDirective *D) {
			//			llvm::outs() << "VisitOMPTaskyieldDirective\n";
			return true;
		}

		//-------------------------------------------------------------------------//
		// Device Constructs
		// 		'target data' Construct
		// 		'target enter data' Construct
		// 		'target exit data' Construct
		// 		'target' Construct
		// 		'target update' Construct
		// 		'declare target' Construct
		// 		'teams' Construct
		// 		'distribute' Construct
		// 		'distribute simd' Construct
		// 		'distrubte parallel for' Construct
		// 		'distribute parallel for simd' Construct
		//-------------------------------------------------------------------------//
		bool VisitOMPTargetDataDirective(OMPTargetDataDirective *D) {
			//			FunctionDecl *FuncDecl = CaptStmtToFunctionDecl(D->getInnermostCapturedStmt());
			//			_CI.getASTContext().getTranslationUnitDecl()->addDecl(FuncDecl);

			return true;
		}

		bool VisitOMPTargetEnterDataDirective(OMPTargetEnterDataDirective *D) {
			//			llvm::outs() << "VisitOMPTargetEnterDataDirective\n";
			return true;
		}

		bool VisitOMPTargetExitDataDirective(OMPTargetExitDataDirective *D) {
			//			llvm::outs() << "VisitOMPTargetExitDataDirective\n";
			return true;
		}

		bool VisitOMPTargetDirective(OMPTargetDirective *D) {
			CapturedStmt *CaptStmt = D->getInnermostCapturedStmt();
			CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
			PrintingPolicy Policy(_CI.getLangOpts());
			Stmt *Body = CaptDecl->getBody();

			std::string Str;
			llvm::raw_string_ostream ss(Str);
			/* 
			 * Function Prototype
			 */
			ss << "\n{\n";
			/* 
			 * Implicit Params
			 */
			for (auto *param : CaptDecl->parameters()) {
				ss << "//";
				param->print(ss, Policy);
				ss << "\n";
			}
			ss << "\n";

			/*
			 * Mapping variables
			 */
			ss << "\n// Initial Memory Management\n";
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
						ss << "int " << created_name.str() << ";\n";
						ss << d->getType().getAsString() << " m_";
						ss << d->getNameAsString() << " = ";
						ss << "CreateOrGetBuffer(";

						if (OMPArraySectionExpr *section = dyn_cast<OMPArraySectionExpr>(expr)) {
							// FIXME: How about pointers?
							std::string startAddress_str;
							llvm::raw_string_ostream startAddress(startAddress_str);
							section->getBase()->printPretty(startAddress, nullptr, Policy);
							startAddress << "[";
							section->getLowerBound()->printPretty(startAddress, nullptr, Policy);
							startAddress << "]";

							ss << "&";
							ss << startAddress.str();
							ss << ", ";
							section->getLength()->printPretty(ss, nullptr, Policy);
							ss << " * sizeof(" << startAddress.str() << "), ";
							ss << (copy ? "1" : "0");
							ss << ", ";
							ss << "&" << created_name.str();
							ss << ");";
							ss << "\n";
							break;
						} else if (DeclRefExpr *var = dyn_cast<DeclRefExpr>(expr)) {
							ss << "&";
							var->printPretty(ss, nullptr, Policy);
							ss << ", sizeof(";
							var->printPretty(ss, nullptr, Policy);
							ss <<"), " << (copy ? "1" : "0");
							ss <<", " << created_name.str();
							ss << ");";
							ss << "\n";
						} else {
							llvm_unreachable("map clause contains only ArraySection or a value");
						}

					}
				}
			}

			ss << "\n";
			/*
			 * Body
			 */
			Body->printPretty(ss, nullptr, Policy, 1);

			ss << "\n// Memory cleanup\n";
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
						ss << "DestroyBuffer(";

						if (OMPArraySectionExpr *section = dyn_cast<OMPArraySectionExpr>(expr)) {
							std::string startAddress_str;
							llvm::raw_string_ostream startAddress(startAddress_str);
							section->getBase()->printPretty(startAddress, nullptr, Policy);
							startAddress << "[";
							section->getLowerBound()->printPretty(startAddress, nullptr, Policy);
							startAddress << "]";

							ss << "&";
							ss << startAddress.str();
							ss << ", ";
							section->getLength()->printPretty(ss, nullptr, Policy);
							ss << " * sizeof(" << startAddress.str() << "), ";
							ss << (copy ? "1" : "0");
							ss << ", ";
							ss << created_name.str();
							ss << ");";
							ss << "\n";
						} else if (DeclRefExpr *var = dyn_cast<DeclRefExpr>(expr)) {
							ss << "&";
							var->printPretty(ss, nullptr, Policy);
							ss << ", sizeof(";
							var->printPretty(ss, nullptr, Policy);
							ss <<"), " << (copy ? "1" : "0");
							ss <<", " << created_name.str();
							ss << ");";
							ss << "\n";
						} else {
							llvm_unreachable("map clause contains only ArraySection or a value");
						}
					}
				}
			}
			ss << "\n}\n";


			SourceManager &SrcMgr = _CI.getSourceManager();
			FileID FID = SrcMgr.getFileID(D->getBeginLoc());
			unsigned LineNo = SrcMgr.getPresumedLineNumber(D->getBeginLoc());
			SourceLocation BeginLoc = SrcMgr.translateLineCol(FID, LineNo, 1);
			SourceLocation EndLoc = D->getAssociatedStmt()->getEndLoc();
			_Rewriter.ReplaceText(SourceRange(BeginLoc, EndLoc), ss.str());

			return false;
		}

		bool VisitOMPTargetUpdateDirective(OMPTargetUpdateDirective *D) {
			//			llvm::outs() << "VisitOMPTargetUpdateDirective\n";
			return true;
		}

		bool VisitOMPDeclareTargetDeclAttr(OMPDeclareTargetDeclAttr *D) {
			//			llvm::outs() << "VisitOMPDeclareTargetDeclAttr\n";
			return true;
		}
		/* 
		   bool VisitOMPEndDeclareTargetDeclAttr(OMPEndDeclareTargetDeclAttr *D) {
		   return true;
		   }
		   */

		bool VisitOMPTeamsDirective(OMPTeamsDirective *D) {
			/*
			 * #pragma omp teams [clause[[,] clause] ... ] new-line
			 * 	structured-block
			 *
			 * clause:
			 * 		num_teams(integer-expression)
			 * 		thread_limit(integer-expression)
			 * 		default(shared | none)
			 * 		private(list)
			 * 		firstprivate(list)
			 * 		shared(list)
			 * 		reduction(reduction-identifier: list)
			 *
			 * NOTE: orphaned-teams region is not allowed in Clang (See OpenMP 4.5 Specification-2.10.7)
			 *
			 */
			CapturedStmt *CaptStmt = D->getInnermostCapturedStmt();
			CapturedDecl *CaptDecl = CaptStmt->getCapturedDecl();
			PrintingPolicy Policy(_CI.getLangOpts());
			Stmt *Body = CaptDecl->getBody();

			std::string Str;
			llvm::raw_string_ostream ss(Str);
			/* 
			 * Function Prototype
			 */
			ss << "\n{\n";
			/* 
			 * Implicit Params
			 */
			for (auto *param : CaptDecl->parameters()) {
				ss << "//";
				param->print(ss, Policy);
				ss << "\n";
			}
			ss << "\n";

			/*
			 * num_teams & thread_limit clauses
			 */
			const OMPNumTeamsClause *NumTeamsClause = D->getSingleClause<OMPNumTeamsClause>();
			const OMPThreadLimitClause *ThreadLimitClause = D->getSingleClause<OMPThreadLimitClause>();

			ss << "\n// team configuration\n";
			if (NumTeamsClause) {
				ss << "PushNumTeams(";
				NumTeamsClause->getNumTeams()->printPretty(ss, nullptr, Policy);
				ss << ");\n";
			} else {
				ss << "PushNumTeams();\n";
			}

			if (ThreadLimitClause) {
				ss << "PushThreadLimit(";
				ThreadLimitClause->getThreadLimit()->printPretty(ss, nullptr, Policy);
				ss << ");\n";
			} else {
				ss << "PushThreadLimit();\n";
			}
			//			ss << 
			ss << "\n";

			/*
			 * Kernel call Body
			 */
			Body->printPretty(ss, nullptr, Policy, 1);


			ss << "\n// team configuration clean up\n";
			if (NumTeamsClause) {
				ss << "PopNumTeams(";
				NumTeamsClause->getNumTeams()->printPretty(ss, nullptr, Policy);
				ss << ");\n";
			} else {
				ss << "PopNumTeams();\n";
			}

			if (ThreadLimitClause) {
				ss << "PopThreadLimit(";
				ThreadLimitClause->getThreadLimit()->printPretty(ss, nullptr, Policy);
				ss << ");\n";
			} else {
				ss << "PopThreadLimit();\n";
			}


			ss << "\n}\n";

//			llvm::outs() << ss.str() << "\n";
			SourceManager &SrcMgr = _CI.getSourceManager();
			FileID FID = SrcMgr.getFileID(D->getBeginLoc());
			unsigned LineNo = SrcMgr.getPresumedLineNumber(D->getBeginLoc());
			SourceLocation BeginLoc = SrcMgr.translateLineCol(FID, LineNo, 1);
			SourceLocation EndLoc = D->getAssociatedStmt()->getEndLoc();
			_Rewriter.ReplaceText(SourceRange(BeginLoc, EndLoc), ss.str());
			return true;
		}

		bool VisitOPMDistributeDirective(OMPDistributeDirective *D) {
			/*
			 * #pragma omp distribute [clause [[,] clause] ... ] new-line
			 * 	for-loops
			 *
			 * clause:
			 * 		private(list)
			 * 		firstprivate(list)
			 * 		lastprivate(list)
			 * 		collapse(n)
			 * 		dist_schedule(kind[, chunk_size])
			 */
			//			llvm::outs() << "VisitOPMDistributeDirective\n";
			return true;
		}

		bool VisitOMPDistributeSimdDirective(OMPDistributeSimdDirective *D) {
			//			llvm::outs() << "VisitOMPDistributeSimdDirective\n";
			return true;
		}

		bool VisitOMPDistributeParallelForDirective(OMPDistributeParallelForDirective *D) {
			/*
			 * #pragma omp distribute parallel for [clause [[,] clause] ... ] new-line
			 * 	for-loops
			 *
			 * clause: any of the clauses accepted by 'distribute' or 'parallel for' directives
			 *
			 */
			//			PrintingPolicy Policy(_CI.getLangOpts());
			//			std::string Str;
			//			llvm::raw_string_ostream ss(Str);
			//
			//			const Stmt *Body = D->getBody();
			//			D->getCond()->printPretty(llvm::outs(), nullptr, Policy);
			//			D->getNumIterations()->printPretty(llvm::outs(), nullptr, Policy);
			//
			//			D->getBody()->printPretty(llvm::outs(), nullptr, Policy);
			//
			//			ss << "void function()\n{";
			//			ss << "";
			//			ss << "\n}\n";
			//

			return true;
		}

		bool VisitOMPDistributeParallelForSimdDirective(OMPDistributeParallelForSimdDirective *D) {
			//			llvm::outs() << "VisitOMPDistributeParallelForSimdDirective\n";
			return true;
		}

		//-------------------------------------------------------------------------//
		// Combined Constructs
		// 		'parallel for' Construct
		// 		'parallel sections' Construct
		// 		'parallel for simd' Construct
		// 		'target parallel' Construct
		// 		'target parallel for' Construct
		// 		'target parallel for simd' Construct
		// 		'target simd' Construct
		// 		'target teams' Construct
		// 		'teams distribute' Construct
		// 		'teams distribute simd' Construct
		// 		'target teams distribute' Construct
		// 		'target teams distribute simd' Construct
		// 		'teams distribute parallel for' Construct
		// 		'target teams distribute parallel for' Construct
		// 		'teams distribute parallel for simd' Construct
		// 		'target teams distribute parallel for simd' Construct
		//-------------------------------------------------------------------------//
		bool VisitOMPParallelForDirective(OMPParallelForDirective *D) {
			SourceManager &SrcMgr = _CI.getSourceManager();
			FileID FID = SrcMgr.getFileID(D->getBeginLoc());
			unsigned LineNo = SrcMgr.getPresumedLineNumber(D->getBeginLoc());
			SourceLocation BeginLoc = SrcMgr.translateLineCol(FID, LineNo, 1);
			SourceLocation EndLoc = D->getAssociatedStmt()->getEndLoc();
			_Rewriter.ReplaceText(SourceRange(BeginLoc, EndLoc), "{ abcd; }");
			return true;
		}

		bool VisitOMPParallelSectionsDirective(OMPParallelSectionsDirective *D) {
			//			llvm::outs() << "VisitOMPParallelSectionsDirective\n";
			return true;
		}

		bool VisitOMPForSimdDirective(OMPForSimdDirective *D) {
			return true;
		}

		bool VisitOMPTargetParallelDirective(OMPTargetParallelDirective *D) {
			//			llvm::outs() << "VisitOMPTargetParallelDirective\n";
			return true;
		}

		bool VisitOMPTargetParallelForDirective(OMPTargetParallelForDirective *D) {
			/*
			 * #pragma omp parallel for [clause [[,] clause] ...] new-line
			 *
			 * clause : any of the clauses accepted by the 'parallel' or 'for' directives
			 * except the 'nowait' clause.
			 *
			 */
			//			llvm::outs() << "VisitOMPTargetParallelForDirective\n";
			return true;
		}

		bool VisitOMPTargetParallelForSimdDirective(OMPTargetParallelForSimdDirective *D) {
			//			llvm::outs() << "VisitOMPTargetParallelForSimdDirective\n";
			return true;
		}

		bool VisitOMPTargetSimdDirective(OMPTargetSimdDirective *D) {
			//			llvm::outs() << "VisitOMPTargetSimdDirective\n";
			return true;
		}

		bool VisitOMPTargetTeamsDirective(OMPTargetTeamsDirective *D) {
			/* #pragma omp target teams */
			//			llvm::outs() << "VisitOMPTargetTeamsDirective\n";
			return true;
		}

		bool VisitOMPTeamsDistributeDirective(OMPTeamsDistributeDirective *D) {
			//			llvm::outs() << "VisitOMPTeamsDistributeDirective\n";
			return true;
		}

		bool VisitOMPTeamsDistributeSimdDirective(OMPTeamsDistributeSimdDirective *D) {
			//			llvm::outs() << "VisitOMPTeamsDistributeSimdDirective\n";
			return true;
		}

		bool VisitOMPTargetTeamsDistributeDirective(OMPTargetTeamsDistributeDirective *D) {
			//			llvm::outs() << "VisitOMPTargetTeamsDistributeDirective\n";
			return true;
		}

		bool VisitOMPTargetTeamsDistributeSimdDirective(OMPTargetTeamsDistributeSimdDirective *D) {
			/* #pragma omp target teams distribute simd */
			//			llvm::outs() << "VisitOMPTargetTeamsDistributeSimdDirective\n";
			return true;
		}

		bool VisitOMPTeamsDistributeParallelForDirective(OMPTeamsDistributeParallelForDirective *D) {
			/*
			 * #pragma omp teams distribute parallel for [clause[[,] clause] ... ] new-line
			 * 	for-loops
			 *
			 * clause: any of the clauses accepted by the 'teams' or 'distribute parallel for' directives
			 *
			 */
			//			llvm::outs() << "VisitOMPTeamsDistributeParallelForDirective\n";
			return true;
		}

		bool VisitOMPTargetTeamsDistributeParallelForDirective(OMPTargetTeamsDistributeParallelForDirective *D) {
			/*
			 * #pragma omp target teams distribute [clause[[,] clause] ... ] new-line
			 * 	for-loops
			 * 
			 * clause: any of clauses accepted by the 'target' or 'teams distribute parallel for' directives
			 *
			 */
			//			CaptStmtToFunctionDecl(D->getInnermostCapturedStmt());
			//			llvm::outs() << "VisitOMPTargetTeamsDistributeParallelForDirective\n";
			return true;
		}

		bool VisitOMPTeamsDistributeParallelForSimdDirective(OMPTeamsDistributeParallelForSimdDirective *D) {
			//			llvm::outs() << "VisitOMPTeamsDistributeParallelForSimdDirective\n";
			return true;
		}

		bool VisitOMPTargetTeamsDistributeParallelForSimdDirective(OMPTargetTeamsDistributeParallelForSimdDirective *D) {
			//			llvm::outs() << "VisitOMPTargetTeamsDistributeParallelForSimdDirective\n";
			return true;
		}

		//-------------------------------------------------------------------------//
		// If Clause
		//-------------------------------------------------------------------------//
		bool VisitOMPIfClause(OMPIfClause *C) {
			//			llvm::outs() << "VisitOMPIfClause\n";
			return true;
		}

		//-------------------------------------------------------------------------//
		// Master and Synchronization Constructs and Clauses
		// 		'master' Construct
		// 		'critical' Construct
		// 		'barrier' Construct
		// 		'taskwait' Construct
		// 		'taskgroup' Construct
		// 		'atomic' Construct
		// 		'flush' Construct
		// 		'ordered' Construct
		// 		'depend' Clause
		//-------------------------------------------------------------------------//

		bool VisitOMPMasterDirective(OMPMasterDirective *D) {
			//			llvm::outs() << "VisitOMPMasterDirective\n";
			return true;
		}

		bool VisitOMPCriticalDirective(OMPCriticalDirective *D) {
			//			llvm::outs() << "VisitOMPCriticalDirective\n";
			return true;
		}

		bool VisitOMPBarrierDirective(OMPBarrierDirective *D) {
			//			llvm::outs() << "VisitOMPBarrierDirective\n";
			return true;
		}

		bool VisitOMPTaskwaitDirective(OMPTaskwaitDirective *D) {
			//			llvm::outs() << "VisitOMPTaskwaitDirective\n";
			return true;
		}

		bool VisitOMPAtomicDirective(OMPAtomicDirective *D) {
			//			llvm::outs() << "VisitOMPAtomicDirective\n";
			return true;
		}

		bool VisitOMPTaskgroupDirective(OMPTaskgroupDirective *D) {
			//			llvm::outs() << "VisitOMPTaskgroupDirective\n";
			return true;
		}

		bool VisitOMPFlushDirective(OMPFlushDirective *D) {
			//			llvm::outs() << "VisitOMPFlushDirective\n";
			return true;
		}

		bool VisitOMPOrderedDirective(OMPOrderedDirective *D) {
			//			llvm::outs() << "VisitOMPOrderedDirective\n";
			return true;
		}

		bool VisitOMPDependClause(OMPDependClause *C) {
			//			llvm::outs() << "VisitOMPDependClause\n";
			return true;
		}

		//-------------------------------------------------------------------------//
		// Cancellation Constructs
		// 		'cancle' Construct
		// 		'cancellation point' Construct
		//-------------------------------------------------------------------------//
		bool VisitOMPCancelDirective(OMPCancelDirective *D) {
			//			llvm::outs() << "VisitOMPCancelDirective\n";
			return true;
		}

		bool VisitOMPCancellationPointDirective(OMPCancellationPointDirective *D) {
			//			llvm::outs() << "VisitOMPCancellationPointDirective\n";
			return true;
		}

		//-------------------------------------------------------------------------//
		// Data Environment
		// 		'threadprivate' Directive
		// 		Data-Sharing Attribute Clauses
		// 				'default' Clause
		// 				'shared' Clause
		// 				'private' Clause
		// 				'firstprivate' Clause
		// 				'lastprivate' Clause
		// 				'reduction' Clause
		// 				'linear' Clause
		// 		Data Copying Clauses
		// 				'copyin' Clause
		// 				'copyprivate' Clause
		// 		Data-mapping Attribute Rules and Clauses
		// 				'map' Clause
		// 				'defaultmap' Clause
		// 		'declare reduction' Directive
		//-------------------------------------------------------------------------//
		bool VisitOMPThreadPrivateDecl(OMPThreadPrivateDecl *D) {
			//			llvm::outs() << "VisitOMPThreadPrivateDecl\n";
			return true;
		}
		bool VisitOMPDefaultClause(OMPDefaultClause *C) {
			//			llvm::outs() << "VisitOMPDefaultClause\n";
			return true;
		}
		bool VisitOMPSharedClause(OMPSharedClause *C) {
			//			llvm::outs() << "VisitOMPSharedClause\n";
			return true;
		}
		bool VisitOMPPrivateClause(OMPPrivateClause *C) {
			//			llvm::outs() << "VisitOMPPrivateClause\n";
			return true;
		}
		bool VisitOMPFirstprivateClause(OMPFirstprivateClause *C) {
			//			llvm::outs() << "VisitOMPFirstprivateClause\n";
			return true;
		}
		bool VisitOMPLastprivateClause(OMPLastprivateClause *C) {
			//			llvm::outs() << "VisitOMPLastprivateClause\n";
			return true;
		}
		bool VisitOMPReductionClause(OMPReductionClause *C) {
			//			llvm::outs() << "VisitOMPReductionClause\n";
			return true;
		}
		bool VisitOMPLinearClause(OMPLinearClause *C) {
			//			llvm::outs() << "VisitOMPLinearClause\n";
			return true;
		}
		bool VisitOMPCopyinClause(OMPCopyinClause *C) {
			//			llvm::outs() << "VisitOMPCopyinClause\n";
			return true;
		}
		bool VisitOMPCopyprivateClause(OMPCopyprivateClause *C) {
			//			llvm::outs() << "VisitOMPCopyprivateClause\n";
			return true;
		}
		bool VisitOMPMapClause(OMPMapClause *C) {
			//			llvm::outs() << "VisitOMPMapClause\n";
			return true;
		}
		bool VisitOMPDefaultmapClause(OMPDefaultmapClause *C) {
			//			llvm::outs() << "VisitOMPDefaultmapClause\n";
			return true;
		}
		bool VisitOMPDeclareReductionDecl(OMPDeclareReductionDecl *D) {
			llvm::outs() << "VisitOMPDeclareReductionDecl\n";
			return true;
		}

		//-------------------------------------------------------------------------//
		// Other Clauses
		//-------------------------------------------------------------------------//
		bool VisitOMPFinalClause(OMPFinalClause *C) {
			//			llvm::outs() << "VisitOMPFinalClause\n";
			return true;
		} 
		bool VisitOMPNumThreadsClause(OMPNumThreadsClause *C) {
			//			llvm::outs() << "VisitOMPNumThreadsClause\n";
			return true;
		} 
		bool VisitOMPSafelenClause(OMPSafelenClause *C) {
			//			llvm::outs() << "VisitOMPSafelenClause\n";
			return true;
		} 
		bool VisitOMPSimdlenClause(OMPSimdlenClause *C) {
			//			llvm::outs() << "VisitOMPSimdlenClause\n";
			return true;
		} 
		bool VisitOMPCollapseClause(OMPCollapseClause *C) {
			//			llvm::outs() << "VisitOMPCollapseClause\n";
			return true;
		} 
		bool VisitOMPAlignedClause(OMPAlignedClause *C) {
			//			llvm::outs() << "VisitOMPAlignedClause\n";
			return true;
		} 
		bool VisitOMPProcBindClause(OMPProcBindClause *C) {
			//			llvm::outs() << "VisitOMPProcBindClause\n";
			return true;
		} 
		bool VisitOMPScheduleClause(OMPScheduleClause *C) {
			//			llvm::outs() << "VisitOMPScheduleClause\n";
			return true;
		} 
		bool VisitOMPOrderedClause(OMPOrderedClause *C) {
			//			llvm::outs() << "VisitOMPOrderedClause\n";
			return true;
		} 
		bool VisitOMPNowaitClause(OMPNowaitClause *C) {
			//			llvm::outs() << "VisitOMPNowaitClause\n";
			return true;
		} 
		bool VisitOMPUntiedClause(OMPUntiedClause *C) {
			//			llvm::outs() << "VisitOMPUntiedClause\n";
			return true;
		} 
		bool VisitOMPMergeableClause(OMPMergeableClause *C) {
			//			llvm::outs() << "VisitOMPMergeableClause\n";
			return true;
		} 
		bool VisitOMPFlushClause(OMPFlushClause *C) {
			//			llvm::outs() << "VisitOMPFlushClause\n";
			return true;
		} 
		bool VisitOMPReadClause(OMPReadClause *C) {
			//			llvm::outs() << "VisitOMPReadClause\n";
			return true;
		} 
		bool VisitOMPWriteClause(OMPWriteClause *C) {
			//			llvm::outs() << "VisitOMPWriteClause\n";
			return true;
		} 
		bool VisitOMPUpdateClause(OMPUpdateClause *C) {
			//			llvm::outs() << "VisitOMPUpdateClause\n";
			return true;
		} 
		bool VisitOMPCaptureClause(OMPCaptureClause *C) {
			//			llvm::outs() << "VisitOMPCaptureClause\n";
			return true;
		} 
		bool VisitOMPSeqCstClause(OMPSeqCstClause *C) {
			//			llvm::outs() << "VisitOMPSeqCstClause\n";
			return true;
		} 
		bool VisitOMPDeviceClause(OMPDeviceClause *C) {
			//			llvm::outs() << "VisitOMPDeviceClause\n";
			return true;
		} 
		bool VisitOMPThreadsClause(OMPThreadsClause *C) {
			//			llvm::outs() << "VisitOMPThreadsClause\n";
			return true;
		} 
		bool VisitOMPSIMDClause(OMPSIMDClause *C) {
			//			llvm::outs() << "VisitOMPSIMDClause\n";
			return true;
		} 
		bool VisitOMPNumTeamsClause(OMPNumTeamsClause *C) {
			//			llvm::outs() << "VisitOMPNumTeamsClause\n";
			return true;
		} 
		bool VisitOMPThreadLimitClause(OMPThreadLimitClause *C) {
			//			llvm::outs() << "VisitOMPThreadLimitClause\n";
			return true;
		} 
		bool VisitOMPPriorityClause(OMPPriorityClause *C) {
			//			llvm::outs() << "VisitOMPPriorityClause\n";
			return true;
		} 
		bool VisitOMPGrainsizeClause(OMPGrainsizeClause *C) {
			//			llvm::outs() << "VisitOMPGrainsizeClause\n";
			return true;
		} 
		bool VisitOMPNogroupClause(OMPNogroupClause *C) {
			//			llvm::outs() << "VisitOMPNogroupClause\n";
			return true;
		} 
		bool VisitOMPNumTasksClause(OMPNumTasksClause *C) {
			//			llvm::outs() << "VisitOMPNumTasksClause\n";
			return true;
		} 
		bool VisitOMPHintClause(OMPHintClause *C) {
			//			llvm::outs() << "VisitOMPHintClause\n";
			return true;
		} 
		bool VisitOMPDistScheduleClause(OMPDistScheduleClause *C) {
			//			llvm::outs() << "VisitOMPDistScheduleClause\n";
			return true;
		} 
		bool VisitOMPToClause(OMPToClause *C) {
			//			llvm::outs() << "VisitOMPToClause\n";
			return true;
		} 
		bool VisitOMPFromClause(OMPFromClause *C) {
			//			llvm::outs() << "VisitOMPFromClause\n";
			return true;
		} 
		bool VisitOMPUseDevicePtrClause(OMPUseDevicePtrClause *C) {
			//			llvm::outs() << "VisitOMPUseDevicePtrClause\n";
			return true;
		} 
		bool VisitOMPIsDevicePtrClause(OMPIsDevicePtrClause *C) {
			//			llvm::outs() << "VisitOMPIsDevicePtrClause\n";
			return true;
		} 
		bool VisitOMPTaskReductionClause(OMPTaskReductionClause *C) {
			//			llvm::outs() << "VisitOMPTaskReductionClause\n";
			return true;
		} 
		bool VisitOMPInReductionClause(OMPInReductionClause *C) {
			//			llvm::outs() << "VisitOMPInReductionClause\n";
			return true;
		} 

	private:
		Rewriter &_Rewriter;
		CompilerInstance &_CI;
};

class OMPConsumer : public ASTConsumer
{
	public:
		OMPConsumer(CompilerInstance &CI, Rewriter &_Rewriter) : Visitor(CI, _Rewriter) {}

		virtual bool HandleTopLevelDecl(DeclGroupRef DR) override {
			for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
				// Traverse the declaration using our AST visitor.
				Visitor.TraverseDecl(*b);
			}
			return true;
		}

	private:
		OMPVisitor Visitor;
};

class OMPAction : public ASTFrontendAction {
	public:
		virtual bool BeginSourceFileAction(CompilerInstance &CI) {
			_Rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
			_Rewriter.getEditBuffer(CI.getSourceManager().getMainFileID());
			return true;
		}

		virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
				CompilerInstance &CI, StringRef InFile) {
			return std::unique_ptr<clang::ASTConsumer>(new OMPConsumer(CI, _Rewriter));
		}

		virtual void EndSourceFileAction() {
			const RewriteBuffer *RewriteBuf = _Rewriter.getRewriteBufferFor(
					getCompilerInstance().getSourceManager().getMainFileID());
			RewriteBuf->write(llvm::outs());
//			_Rewriter.overwriteChangedFiles();
		}

	private:
		Rewriter _Rewriter;
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
	OMPAction Act;
	CI.ExecuteAction(Act);
	return 0;
}

