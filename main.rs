use std::env;
use std::fs;

// Token definitions
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Module, EndModule, Fun, EndFun, If, EndIf, Else, EndElse,
    While, EndWhile, For, EndFor, Return, EndReturn, Break,
    Int, String, Bool,
    IntLiteral(i32),
    StringLiteral(std::string::String),
    CharLiteral(char),
    BoolLiteral(bool),
    Identifier(std::string::String),
    Arrow, Plus, Minus, Multiply, Percent,
    Greater, Less, GreaterEq, LessEq, Equal, NotEqual, Increment,
    LeftBracket, RightBracket, LeftParen, RightParen, Comma,
    In, Out,
    Newline, Eof,
}

// Lexer (same as before)
struct Lexer {
    input: Vec<char>,
    position: usize,
}

impl Lexer {
    fn new(input: std::string::String) -> Self {
        Lexer { input: input.chars().collect(), position: 0 }
    }
    
    fn current_char(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }
    
    fn peek_char(&self, offset: usize) -> Option<char> {
        self.input.get(self.position + offset).copied()
    }
    
    fn advance(&mut self) {
        self.position += 1;
    }
    
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch == ' ' || ch == '\t' || ch == '\r' {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    fn skip_comment(&mut self) {
        if self.current_char() == Some('/') && self.peek_char(1) == Some('/') {
            while let Some(ch) = self.current_char() {
                if ch == '\n' {
                    break;
                }
                self.advance();
            }
        }
    }
    
    fn read_identifier(&mut self) -> std::string::String {
        let mut result = std::string::String::new();
        while let Some(ch) = self.current_char() {
            if ch.is_alphanumeric() || ch == '_' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        result
    }
    
    fn read_number(&mut self) -> i32 {
        let mut result = std::string::String::new();
        while let Some(ch) = self.current_char() {
            if ch.is_numeric() {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        result.parse().unwrap_or(0)
    }
    
    fn read_string(&mut self) -> std::string::String {
        let mut result = std::string::String::new();
        self.advance(); // skip opening '
        while let Some(ch) = self.current_char() {
            if ch == '\'' {
                self.advance();
                break;
            }
            if ch == '\\' {
                self.advance();
                if let Some(next) = self.current_char() {
                    match next {
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        '\\' => result.push('\\'),
                        '\'' => result.push('\''),
                        _ => result.push(next),
                    }
                    self.advance();
                }
            } else {
                result.push(ch);
                self.advance();
            }
        }
        result
    }
    
    fn next_token(&mut self) -> Token {
        loop {
            self.skip_whitespace();
            if self.current_char() == Some('/') && self.peek_char(1) == Some('/') {
                self.skip_comment();
                continue;
            }
            break;
        }
        
        match self.current_char() {
            None => Token::Eof,
            Some('\n') => { self.advance(); Token::Newline }
            Some('{') => {
                self.advance();
                let ident = self.read_identifier();
                if self.current_char() == Some('}') {
                    self.advance();
                }
                match ident.as_str() {
                    "module" => Token::Module,
                    "/module" => Token::EndModule,
                    "fun" => Token::Fun,
                    "/fun" => Token::EndFun,
                    "if" => Token::If,
                    "/if" => Token::EndIf,
                    "else" => Token::Else,
                    "/else" => Token::EndElse,
                    "while" => Token::While,
                    "/while" => Token::EndWhile,
                    "for" => Token::For,
                    "/for" => Token::EndFor,
                    "return" => Token::Return,
                    "/return" => Token::EndReturn,
                    _ => Token::Identifier(ident),
                }
            }
            Some('-') => {
                self.advance();
                if self.current_char() == Some('>') {
                    self.advance();
                    Token::Arrow
                } else {
                    Token::Minus
                }
            }
            Some('+') => {
                self.advance();
                if self.current_char() == Some('+') {
                    self.advance();
                    Token::Increment
                } else {
                    Token::Plus
                }
            }
            Some('*') => { self.advance(); Token::Multiply }
            Some('%') => { self.advance(); Token::Percent }
            Some('>') => {
                self.advance();
                if self.current_char() == Some('=') {
                    self.advance();
                    Token::GreaterEq
                } else {
                    Token::Greater
                }
            }
            Some('<') => {
                self.advance();
                if self.current_char() == Some('=') {
                    self.advance();
                    Token::LessEq
                } else {
                    Token::Less
                }
            }
            Some('!') => {
                self.advance();
                if self.current_char() == Some('=') {
                    self.advance();
                    Token::NotEqual
                } else {
                    Token::Identifier("!".to_string())
                }
            }
            Some('=') => {
                self.advance();
                if self.current_char() == Some('=') {
                    self.advance();
                    Token::Equal
                } else {
                    Token::Identifier("=".to_string())
                }
            }
            Some('[') => { self.advance(); Token::LeftBracket }
            Some(']') => { self.advance(); Token::RightBracket }
            Some('(') => { self.advance(); Token::LeftParen }
            Some(')') => { self.advance(); Token::RightParen }
            Some(',') => { self.advance(); Token::Comma }
            Some('\'') => {
                let s = self.read_string();
                if s.len() == 1 {
                    Token::CharLiteral(s.chars().next().unwrap())
                } else {
                    Token::StringLiteral(s)
                }
            }
            Some(ch) if ch.is_numeric() => {
                let num = self.read_number();
                Token::IntLiteral(num)
            }
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let ident = self.read_identifier();
                match ident.as_str() {
                    "int" => Token::Int,
                    "string" => Token::String,
                    "bool" => Token::Bool,
                    "true" => Token::BoolLiteral(true),
                    "false" => Token::BoolLiteral(false),
                    "in" => Token::In,
                    "out" => Token::Out,
                    "break" => Token::Break,
                    _ => Token::Identifier(ident),
                }
            }
            Some(_) => {
                self.advance();
                self.next_token()
            }
        }
    }
    
    fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            if token == Token::Eof {
                tokens.push(token);
                break;
            }
            if token != Token::Newline {
                tokens.push(token);
            }
        }
        tokens
    }
}

// AST - Simplified with pipeline support
#[derive(Debug, Clone)]
enum Expr {
    IntLiteral(i32),
    StringLiteral(std::string::String),
    CharLiteral(char),
    BoolLiteral(bool),
    Identifier(std::string::String),
    ArrayAccess(std::string::String, Box<Expr>),
    BinaryOp(Box<Expr>, std::string::String, Box<Expr>),
}

#[derive(Debug, Clone)]
struct VarDecl {
    var_type: std::string::String,
    names: Vec<(std::string::String, Option<Expr>)>, // name, optional init value
    is_array: bool,
    array_size: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
enum Statement {
    VarDecl(VarDecl),
    Assignment(Expr, Expr), // left, right (reversed: right -> left)
    Pipeline(Vec<PipelineStep>),
    If(Expr, Vec<Statement>, Vec<Statement>),
    While(Expr, Vec<Statement>),
    For(std::string::String, Expr, std::string::String, Expr, Vec<Statement>),
    Return(Option<Expr>),
    Break,
}

#[derive(Debug, Clone)]
enum PipelineStep {
    Input,
    Variables(Vec<std::string::String>),
    FunctionCall(std::string::String),
    Output,
    Expr(Expr),
}

#[derive(Debug, Clone)]
struct Function {
    name: std::string::String,
    params: Vec<(std::string::String, std::string::String)>,
    return_type: Option<std::string::String>,
    body: Vec<Statement>,
}

#[derive(Debug, Clone)]
struct Module {
    _name: std::string::String,
    functions: Vec<Function>,
}

// Parser with pipeline support
struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, position: 0 }
    }
    
    fn current(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::Eof)
    }
    
    fn advance(&mut self) {
        self.position += 1;
    }
    
    fn parse_module(&mut self) -> Module {
        // Skip to {module}
        while !matches!(self.current(), Token::Module | Token::Eof) {
            self.advance();
        }
        
        if matches!(self.current(), Token::Eof) {
            return Module { _name: "main".to_string(), functions: Vec::new() };
        }
        
        self.advance(); // skip {module}
        
        let name = if let Token::Identifier(n) = self.current() {
            let name = n.clone();
            self.advance();
            name
        } else {
            "main".to_string()
        };
        
        let mut functions = Vec::new();
        while !matches!(self.current(), Token::EndModule | Token::Eof) {
            if matches!(self.current(), Token::Fun) {
                functions.push(self.parse_function());
            } else {
                self.advance();
            }
        }
        
        Module { _name: name, functions }
    }
    
    fn parse_function(&mut self) -> Function {
        self.advance(); // skip {fun}
        
        let mut params = Vec::new();
        let mut name = std::string::String::new();
        let mut return_type = None;
        
        // Collect tokens until arrow or special pattern
        let mut tokens_before_arrow = Vec::new();
        
        while !matches!(self.current(), Token::EndFun | Token::Eof) {
            if matches!(self.current(), Token::Arrow) {
                self.advance();
                if let Token::Identifier(n) = self.current() {
                    name = n.clone();
                    self.advance();
                }
                if matches!(self.current(), Token::Int | Token::String | Token::Bool) {
                    return_type = Some(self.parse_type());
                }
                break;
            }
            
            if let Token::Identifier(ref id) = self.current() {
                if id == "main" {
                    name = "main".to_string();
                    self.advance();
                    if matches!(self.current(), Token::Int) {
                        return_type = Some("int".to_string());
                        self.advance();
                    }
                    break;
                }
            }
            
            tokens_before_arrow.push(self.current().clone());
            self.advance();
        }
        
        // Parse parameters
        let mut i = 0;
        while i < tokens_before_arrow.len() {
            if matches!(&tokens_before_arrow[i], Token::Int | Token::String | Token::Bool) {
                let ptype = match &tokens_before_arrow[i] {
                    Token::Int => "int",
                    Token::String => "string",
                    Token::Bool => "bool",
                    _ => "int",
                }.to_string();
                i += 1;
                
                if i < tokens_before_arrow.len() {
                    if let Token::Identifier(pname) = &tokens_before_arrow[i] {
                        let pn = pname.clone();
                        i += 1;
                        
                        if i < tokens_before_arrow.len() && matches!(&tokens_before_arrow[i], Token::LeftBracket) {
                            i += 1;
                            if i < tokens_before_arrow.len() && matches!(&tokens_before_arrow[i], Token::RightBracket) {
                                i += 1;
                            }
                            params.push((format!("{}[]", ptype), pn));
                        } else {
                            params.push((ptype, pn));
                        }
                        
                        if i < tokens_before_arrow.len() && matches!(&tokens_before_arrow[i], Token::Comma) {
                            i += 1;
                        }
                    }
                }
            } else {
                i += 1;
            }
        }
        
        let body = self.parse_body();
        Function { name, params, return_type, body }
    }
    
    fn parse_type(&mut self) -> std::string::String {
        let t = match self.current() {
            Token::Int => "int",
            Token::String => "string",
            Token::Bool => "bool",
            _ => "void",
        }.to_string();
        if t != "void" {
            self.advance();
        }
        t
    }
    
    fn parse_body(&mut self) -> Vec<Statement> {
        let mut stmts = Vec::new();
        while !matches!(self.current(), Token::EndFun | Token::Eof) {
            if let Some(stmt) = self.parse_statement() {
                stmts.push(stmt);
            } else {
                break; // Stop if we can't parse a statement
            }
        }
        if matches!(self.current(), Token::EndFun) {
            self.advance();
        }
        stmts
    }
    
    fn parse_statement(&mut self) -> Option<Statement> {
        match self.current() {
            Token::If => Some(self.parse_if()),
            Token::While => Some(self.parse_while()),
            Token::For => Some(self.parse_for()),
            Token::Return => Some(self.parse_return()),
            Token::Break => { self.advance(); Some(Statement::Break) }
            Token::Int | Token::String | Token::Bool => Some(self.parse_var_decl()),
            Token::In | Token::Identifier(_) | Token::IntLiteral(_) | Token::StringLiteral(_) | Token::CharLiteral(_) | Token::BoolLiteral(_) => {
                self.parse_pipeline_or_assignment()
            }
            _ => None,
        }
    }
    
    fn parse_var_decl(&mut self) -> Statement {
        let vtype = self.parse_type();
        let mut names = Vec::new();
        let mut is_array = false;
        let mut array_size = None;
        
        loop {
            if let Token::Identifier(name) = self.current() {
                let n = name.clone();
                self.advance();
                
                let mut init = None;
                
                if matches!(self.current(), Token::LeftBracket) {
                    self.advance();
                    is_array = true;
                    if !matches!(self.current(), Token::RightBracket) {
                        array_size = Some(Box::new(self.parse_expr()));
                    }
                    if matches!(self.current(), Token::RightBracket) {
                        self.advance();
                    }
                }
                
                if matches!(self.current(), Token::LeftParen) {
                    self.advance();
                    init = Some(self.parse_expr());
                    if matches!(self.current(), Token::RightParen) {
                        self.advance();
                    }
                }
                
                names.push((n, init));
                
                if matches!(self.current(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        Statement::VarDecl(VarDecl { var_type: vtype, names, is_array, array_size })
    }
    
    fn parse_if(&mut self) -> Statement {
        self.advance(); // skip {if}
        let cond = self.parse_expr();
        let mut then_body = Vec::new();
        let mut else_body = Vec::new();
        
        while !matches!(self.current(), Token::EndIf | Token::Else | Token::Eof) {
            if let Some(stmt) = self.parse_statement() {
                then_body.push(stmt);
            } else {
                break; // Stop when we hit something we can't parse
            }
        }
        
        if matches!(self.current(), Token::Else) {
            self.advance();
            while !matches!(self.current(), Token::EndElse | Token::Eof) {
                if let Some(stmt) = self.parse_statement() {
                    else_body.push(stmt);
                } else {
                    break;
                }
            }
            if matches!(self.current(), Token::EndElse) {
                self.advance();
            }
        }
        
        if matches!(self.current(), Token::EndIf) {
            self.advance();
        }
        
        Statement::If(cond, then_body, else_body)
    }
    
    fn parse_while(&mut self) -> Statement {
        self.advance(); // skip {while}
        let cond = self.parse_expr();
        let mut body = Vec::new();
        
        while !matches!(self.current(), Token::EndWhile | Token::Eof) {
            if let Some(stmt) = self.parse_statement() {
                body.push(stmt);
            } else {
                break;
            }
        }
        
        if matches!(self.current(), Token::EndWhile) {
            self.advance();
        }
        
        Statement::While(cond, body)
    }
    
    fn parse_for(&mut self) -> Statement {
        self.advance(); // skip {for}
        
        let var_name = if let Token::Identifier(n) = self.current() {
            let name = n.clone();
            self.advance();
            name
        } else {
            "i".to_string()
        };
        
        let mut init = Expr::IntLiteral(0);
        if matches!(self.current(), Token::LeftParen) {
            self.advance();
            init = self.parse_expr();
            if matches!(self.current(), Token::RightParen) {
                self.advance();
            }
        }
        
        let _step = if matches!(self.current(), Token::Increment) {
            self.advance();
            "++"
        } else {
            "++"
        }.to_string();
        
        let op = if matches!(self.current(), Token::Less) {
            self.advance();
            "<"
        } else if matches!(self.current(), Token::Greater) {
            self.advance();
            ">"
        } else {
            "<"
        }.to_string();
        
        let limit = self.parse_expr();
        let cond = Expr::BinaryOp(Box::new(Expr::Identifier(var_name.clone())), op.clone(), Box::new(limit));
        
        let mut body = Vec::new();
        while !matches!(self.current(), Token::EndFor | Token::Eof) {
            if let Some(stmt) = self.parse_statement() {
                body.push(stmt);
            } else {
                break;
            }
        }
        
        if matches!(self.current(), Token::EndFor) {
            self.advance();
        }
        
        Statement::For(var_name, init, "++".to_string(), cond, body)
    }
    
    fn parse_return(&mut self) -> Statement {
        self.advance(); // skip {return}
        let val = if !matches!(self.current(), Token::EndReturn) {
            Some(self.parse_expr())
        } else {
            None
        };
        if matches!(self.current(), Token::EndReturn) {
            self.advance();
        }
        Statement::Return(val)
    }
    
    fn parse_pipeline_or_assignment(&mut self) -> Option<Statement> {
        let mut steps = Vec::new();
        let mut exprs = Vec::new();
        
        loop {
            if matches!(self.current(), Token::EndFun | Token::EndIf | Token::EndElse | Token::EndWhile | Token::EndFor | Token::EndReturn | Token::Eof) {
                break;
            }
            if matches!(self.current(), Token::If | Token::While | Token::For | Token::Return | Token::Break | Token::Int | Token::String | Token::Bool) {
                break;
            }
            
            if matches!(self.current(), Token::In) {
                steps.push(PipelineStep::Input);
                self.advance();
            } else if matches!(self.current(), Token::Out) {
                steps.push(PipelineStep::Output);
                self.advance();
            } else {
                exprs.push(self.parse_expr());
            }
            
            if matches!(self.current(), Token::Arrow) {
                self.advance();
            } else if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        // Analyze the pattern
        if steps.len() > 0 {
            // It's a pipeline with input/output
            return Some(Statement::Pipeline(self.build_pipeline_steps(steps, exprs)));
        }
        
        // Simple assignment: expr -> var
        if exprs.len() == 2 {
            return Some(Statement::Assignment(exprs[1].clone(), exprs[0].clone()));
        }
        
        // Single expression - skip it to avoid invalid code
        None
    }
    
    fn build_pipeline_steps(&self, steps: Vec<PipelineStep>, exprs: Vec<Expr>) -> Vec<PipelineStep> {
        let mut result = Vec::new();
        let mut expr_idx = 0;
        
        for step in steps {
            result.push(step.clone());
            
            // Add expressions between pipeline tokens
            while expr_idx < exprs.len() {
                if expr_idx + 1 < result.len() {
                    break;
                }
                
                if let Expr::Identifier(ref name) = exprs[expr_idx] {
                    if name.chars().next().unwrap().is_uppercase() {
                        result.push(PipelineStep::FunctionCall(name.clone()));
                    } else {
                        result.push(PipelineStep::Variables(vec![name.clone()]));
                    }
                } else {
                    result.push(PipelineStep::Expr(exprs[expr_idx].clone()));
                }
                expr_idx += 1;
            }
        }
        
        // Add remaining expressions
        while expr_idx < exprs.len() {
            if let Expr::Identifier(ref name) = exprs[expr_idx] {
                if name.chars().next().unwrap().is_uppercase() {
                    result.push(PipelineStep::FunctionCall(name.clone()));
                } else {
                    result.push(PipelineStep::Variables(vec![name.clone()]));
                }
            } else {
                result.push(PipelineStep::Expr(exprs[expr_idx].clone()));
            }
            expr_idx += 1;
        }
        
        result
    }
    
    fn parse_expr(&mut self) -> Expr {
        self.parse_comparison()
    }
    
    fn parse_comparison(&mut self) -> Expr {
        let mut left = self.parse_additive();
        
        while let Some(op) = match self.current() {
            Token::Greater => Some(">"),
            Token::Less => Some("<"),
            Token::GreaterEq => Some(">="),
            Token::LessEq => Some("<="),
            Token::Equal => Some("=="),
            Token::NotEqual => Some("!="),
            _ => None,
        } {
            self.advance();
            let right = self.parse_additive();
            left = Expr::BinaryOp(Box::new(left), op.to_string(), Box::new(right));
        }
        
        left
    }
    
    fn parse_additive(&mut self) -> Expr {
        let mut left = self.parse_multiplicative();
        
        while let Some(op) = match self.current() {
            Token::Plus => Some("+"),
            Token::Minus => Some("-"),
            _ => None,
        } {
            self.advance();
            let right = self.parse_multiplicative();
            left = Expr::BinaryOp(Box::new(left), op.to_string(), Box::new(right));
        }
        
        left
    }
    
    fn parse_multiplicative(&mut self) -> Expr {
        let mut left = self.parse_primary();
        
        while let Some(op) = match self.current() {
            Token::Multiply => Some("*"),
            Token::Percent => Some("%"),
            _ => None,
        } {
            self.advance();
            let right = self.parse_primary();
            left = Expr::BinaryOp(Box::new(left), op.to_string(), Box::new(right));
        }
        
        left
    }
    
    fn parse_primary(&mut self) -> Expr {
        match self.current().clone() {
            Token::IntLiteral(n) => { self.advance(); Expr::IntLiteral(n) }
            Token::StringLiteral(s) => { self.advance(); Expr::StringLiteral(s) }
            Token::CharLiteral(c) => { self.advance(); Expr::CharLiteral(c) }
            Token::BoolLiteral(b) => { self.advance(); Expr::BoolLiteral(b) }
            Token::Identifier(name) => {
                self.advance();
                if matches!(self.current(), Token::LeftBracket) {
                    self.advance();
                    let idx = self.parse_expr();
                    if matches!(self.current(), Token::RightBracket) {
                        self.advance();
                    }
                    Expr::ArrayAccess(name, Box::new(idx))
                } else {
                    Expr::Identifier(name)
                }
            }
            Token::LeftParen => {
                self.advance();
                let e = self.parse_expr();
                if matches!(self.current(), Token::RightParen) {
                    self.advance();
                }
                e
            }
            _ => Expr::IntLiteral(0),
        }
    }
}

// Code Generator
struct CodeGen {
    indent: usize,
}

impl CodeGen {
    fn new() -> Self {
        CodeGen { indent: 0 }
    }
    
    fn ind(&self) -> std::string::String {
        "    ".repeat(self.indent)
    }
    
    fn generate(&mut self, module: &Module) -> std::string::String {
        let mut code = std::string::String::from("use std::io::{self, BufRead};\n\n");
        
        // Add helper function for reading integers
        code.push_str("fn read_int() -> i32 {\n");
        code.push_str("    let mut line = String::new();\n");
        code.push_str("    io::stdin().lock().read_line(&mut line).expect(\"Failed to read\");\n");
        code.push_str("    line.trim().parse().unwrap_or(0)\n");
        code.push_str("}\n\n");
        
        for func in &module.functions {
            code.push_str(&self.gen_function(func));
            code.push_str("\n");
        }
        
        code
    }
    
    fn gen_function(&mut self, func: &Function) -> std::string::String {
        let mut code = std::string::String::new();
        
        if func.name == "main" {
            code.push_str("fn main() {\n");
        } else {
            code.push_str(&format!("fn {}(", func.name));
            for (i, (ptype, pname)) in func.params.iter().enumerate() {
                if i > 0 { code.push_str(", "); }
                code.push_str(&format!("{}: {}", pname, self.map_type(ptype)));
            }
            let rtype = func.return_type.as_deref().unwrap_or("()");
            code.push_str(&format!(") -> {} {{\n", self.map_type(rtype)));
        }
        
        self.indent += 1;
        for stmt in &func.body {
            code.push_str(&self.gen_stmt(stmt));
        }
        self.indent -= 1;
        
        code.push_str("}\n");
        code
    }
    
    fn map_type(&self, t: &str) -> std::string::String {
        match t {
            "int" => "i32",
            "string" => "String",
            "bool" => "bool",
            "string[]" => "Vec<String>",
            _ if t.ends_with("[]") => {
                let base = &t[..t.len()-2];
                return format!("Vec<{}>", self.map_type(base));
            }
            _ => t,
        }.to_string()
    }
    
    fn gen_stmt(&mut self, stmt: &Statement) -> std::string::String {
        match stmt {
            Statement::VarDecl(decl) => self.gen_var_decl(decl),
            Statement::Assignment(left, right) => {
                format!("{}{};\n", self.ind(), self.gen_assign(left, right))
            }
            Statement::Pipeline(steps) => self.gen_pipeline(steps),
            Statement::If(cond, then_b, else_b) => self.gen_if(cond, then_b, else_b),
            Statement::While(cond, body) => self.gen_while(cond, body),
            Statement::For(var, init, _step, cond, body) => self.gen_for(var, init, cond, body),
            Statement::Return(val) => {
                if let Some(v) = val {
                    format!("{}return {};\n", self.ind(), self.gen_expr(v))
                } else {
                    format!("{}return;\n", self.ind())
                }
            }
            Statement::Break => format!("{}break;\n", self.ind()),
        }
    }
    
    fn gen_var_decl(&mut self, decl: &VarDecl) -> std::string::String {
        let mut code = std::string::String::new();
        let rtype = self.map_type(&decl.var_type);
        
        if decl.is_array {
            for (name, _init) in &decl.names {
                if let Some(size) = &decl.array_size {
                    let sz = self.gen_expr(size);
                    code.push_str(&format!("{}let mut {} = vec![{}::default(); {} as usize];\n",
                        self.ind(), name, rtype, sz));
                } else {
                    code.push_str(&format!("{}let mut {}: Vec<{}> = Vec::new();\n",
                        self.ind(), name, rtype));
                }
            }
        } else {
            for (name, init) in &decl.names {
                if let Some(i) = init {
                    code.push_str(&format!("{}let mut {}: {} = {};\n",
                        self.ind(), name, rtype, self.gen_expr(i)));
                } else {
                    code.push_str(&format!("{}let mut {}: {};\n",
                        self.ind(), name, rtype));
                }
            }
        }
        
        code
    }
    
    fn gen_assign(&self, left: &Expr, right: &Expr) -> std::string::String {
        format!("{} = {}", self.gen_expr(left), self.gen_expr(right))
    }
    
    fn gen_pipeline(&mut self, steps: &[PipelineStep]) -> std::string::String {
        let mut code = std::string::String::new();
        
        // Simple heuristic: in -> vars -> func -> out
        // Generate: vars = read; result = func(vars); print(result);
        
        let has_input = steps.iter().any(|s| matches!(s, PipelineStep::Input));
        let has_output = steps.iter().any(|s| matches!(s, PipelineStep::Output));
        
        let mut vars = Vec::new();
        let mut func_name = None;
        
        for step in steps {
            match step {
                PipelineStep::Variables(v) => vars.extend(v.clone()),
                PipelineStep::FunctionCall(f) => func_name = Some(f.clone()),
                _ => {}
            }
        }
        
        if has_input {
            for var in &vars {
                code.push_str(&format!("{}{} = read_int();\n", self.ind(), var));
            }
        }
        
        if let Some(fname) = func_name {
            if has_output {
                let args = vars.join(", ");
                code.push_str(&format!("{}println!(\"{{}}\", {}({}));\n", self.ind(), fname, args));
            } else {
                let args = vars.join(", ");
                code.push_str(&format!("{}{}({});\n", self.ind(), fname, args));
            }
        } else if has_output && !vars.is_empty() {
            code.push_str(&format!("{}println!(\"{{}}\", {});\n", self.ind(), vars[0]));
        }
        
        code
    }
    
    fn gen_if(&mut self, cond: &Expr, then_b: &[Statement], else_b: &[Statement]) -> std::string::String {
        let mut code = format!("{}if {} {{\n", self.ind(), self.gen_expr(cond));
        
        self.indent += 1;
        for stmt in then_b {
            code.push_str(&self.gen_stmt(stmt));
        }
        self.indent -= 1;
        
        if !else_b.is_empty() {
            code.push_str(&format!("{}}} else {{\n", self.ind()));
            self.indent += 1;
            for stmt in else_b {
                code.push_str(&self.gen_stmt(stmt));
            }
            self.indent -= 1;
        }
        
        code.push_str(&format!("{}}}\n", self.ind()));
        code
    }
    
    fn gen_while(&mut self, cond: &Expr, body: &[Statement]) -> std::string::String {
        let mut code = format!("{}while {} {{\n", self.ind(), self.gen_expr(cond));
        
        self.indent += 1;
        for stmt in body {
            code.push_str(&self.gen_stmt(stmt));
        }
        self.indent -= 1;
        
        code.push_str(&format!("{}}}\n", self.ind()));
        code
    }
    
    fn gen_for(&mut self, var: &str, init: &Expr, cond: &Expr, body: &[Statement]) -> std::string::String {
        // Extract limit from condition
        if let Expr::BinaryOp(_, op, limit) = cond {
            let init_val = self.gen_expr(init);
            let limit_val = self.gen_expr(limit);
            
            let mut code = format!("{}for {} in {}..{} {{\n", self.ind(), var, init_val, limit_val);
            
            self.indent += 1;
            for stmt in body {
                code.push_str(&self.gen_stmt(stmt));
            }
            self.indent -= 1;
            
            code.push_str(&format!("{}}}\n", self.ind()));
            return code;
        }
        
        std::string::String::new()
    }
    
    fn gen_expr(&self, expr: &Expr) -> std::string::String {
        match expr {
            Expr::IntLiteral(n) => n.to_string(),
            Expr::StringLiteral(s) => format!("\"{}\"", s),
            Expr::CharLiteral(c) => format!("'{}'", c),
            Expr::BoolLiteral(b) => b.to_string(),
            Expr::Identifier(name) => name.clone(),
            Expr::ArrayAccess(arr, idx) => format!("{}[{} as usize]", arr, self.gen_expr(idx)),
            Expr::BinaryOp(left, op, right) => {
                format!("({} {} {})", self.gen_expr(left), op, self.gen_expr(right))
            }
        }
    }
}

fn main() {
    let args: Vec<std::string::String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <aura_file>", args[0]);
        std::process::exit(1);
    }
    
    let source = fs::read_to_string(&args[1]).expect("Failed to read file");
    
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();
    
    let mut parser = Parser::new(tokens);
    let module = parser.parse_module();
    
    let mut codegen = CodeGen::new();
    let rust_code = codegen.generate(&module);
    
    println!("{}", rust_code);
}
