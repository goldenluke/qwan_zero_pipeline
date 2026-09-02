/*
    Atividade Avaliativa - A2
    Disciplina de Estruturas de Dados

    Aluno1: João Victor Barroso de Machado Carvalhos
    Aluno2: Kauê Marin
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PATH 100
#define MAX_NOME 30

/**
 * Você deverá completar a definição dos TADS (Movimentacao, Conta, Cliente e Banco)
*/

typedef struct _mov *ptrMovimentacao;
typedef struct _mov {
    int tipo;
    int idClienteDest;
    int idClienteOrig;
    float valor;
    struct _mov *prox;
} Movimentacao;

typedef struct _conta {
    int numero;
    float saldoInicial;   
    float saldo;
    ptrMovimentacao movimentacoes;
} Conta;

typedef struct _cliente *ptrCliente;
typedef struct _cliente {
    
    int id;
    Conta conta;
    struct _cliente *prox;
    
} Cliente;

typedef struct _banco {
    char nome[MAX_NOME];
    ptrCliente clientes;
} Banco;


/**
    Função para criacao de uma nova movimentacao
    Você deverá implementar as funcionalidades desta função
*/
ptrMovimentacao criarNovaMovimentacao(short int tipo, float valor, int idClienteOrig, int idClienteDest) {
    ptrMovimentacao novaMov = (ptrMovimentacao) malloc(sizeof(Movimentacao));

    if (novaMov != NULL) {
        novaMov->tipo = (int) tipo;
        novaMov->valor = valor;
        novaMov->idClienteOrig = idClienteOrig;
        if (tipo != 2) {
            novaMov->idClienteDest = 0;
        } else {
            novaMov->idClienteDest = idClienteDest;
        }
        novaMov->prox = NULL;
    }

    return novaMov;
}
/**
    Função que realiza a busca de um determinado cliente.
    Esta função deverá retornar NULL caso cliente não esteja na lista.
    Você deverá implementar as funcionalidades desta função
*/
Cliente* buscarCliente(Banco *b, int idCliente) {
    if(b == NULL){
        return NULL;
    }

    ptrCliente atual = b->clientes;

    while (atual != NULL) {
        if (atual->id == idCliente) {
            return atual;
        }
        atual = atual->prox;
    }

    return NULL; 
}

/**
    Função que tem por objetivo realizar uma movimentação de saque: Adição de uma movimentação no cliente c
    Você deverá implementar as funcionalidades desta função
*/
void realizarSaque(Cliente *c, Movimentacao* saque) {
    if (c == NULL || saque == NULL) return;

    c->conta.saldo -= saque->valor;

    if (c->conta.movimentacoes == NULL) {
        c->conta.movimentacoes = saque;
    } else {
        ptrMovimentacao atual = c->conta.movimentacoes;
        while (atual->prox != NULL) {
            atual = atual->prox;
        }
        atual->prox = saque;
    }
}

/**
    Função que tem por objetivo realizar uma movimentação de depósito: Adição de uma movimentação no cliente c
    Você deverá implementar as funcionalidades desta função
*/
void realizarDeposito(Cliente *c, Movimentacao* dep) {
    if (c == NULL || dep == NULL) return;

    c->conta.saldo += dep->valor;

    if (c->conta.movimentacoes == NULL) {
        c->conta.movimentacoes = dep;
    } else {
        ptrMovimentacao atual = c->conta.movimentacoes;
        while (atual->prox != NULL) {
            atual = atual->prox;
        }
        atual->prox = dep;
    }
}

/**
    Função que tem por objetivo a realização de transferencia: orig -> dest
    Você deverá implementar as funcionalidades desta função
*/
void realizarTransferencia(Cliente *orig, Cliente *dest, float valor) {
    if (orig == NULL || dest == NULL || valor <= 0) return;

    orig->conta.saldo -= valor;
    dest->conta.saldo += valor;

    ptrMovimentacao movOrig = criarNovaMovimentacao(2, valor, orig->id, dest->id);

    ptrMovimentacao movDest = criarNovaMovimentacao(2, valor, orig->id, dest->id);

    if (orig->conta.movimentacoes == NULL) {
        orig->conta.movimentacoes = movOrig;
    } else {
        ptrMovimentacao atual = orig->conta.movimentacoes;
        while (atual->prox != NULL) {
            atual = atual->prox;
        }
        atual->prox = movOrig;
    }

    if (dest->conta.movimentacoes == NULL) {
        dest->conta.movimentacoes = movDest;
    } else {
        ptrMovimentacao atual = dest->conta.movimentacoes;
        while (atual->prox != NULL) {
            atual = atual->prox;
        }
        atual->prox = movDest;
    }
}


/**
    Função que tem por objetivo a adição de um cliente no banco de forma ordenada
    Você deverá implementar as funcionalidades desta função
*/
void adicionarCliente(Banco *b, Cliente *c) { 
    if (b == NULL || c == NULL) return;

    if (b->clientes == NULL || c->id < b->clientes->id) { 
        c->prox = b->clientes; 
        b->clientes = c; 
        return; 
    } 

    ptrCliente anterior = b->clientes; 
    ptrCliente atual = b->clientes->prox; 

    while (atual != NULL && atual->id < c->id) { 
        anterior = atual; 
        atual = atual->prox; 
    } 

    c->prox = atual; 
    anterior->prox = c;
}
/**
    Função que tem por objetivo a adição de um novo cliente com seus dados iniciais.
    Lembre-se que um cliente deverá possuir uma conta e esta conta possui uma lista de movimentações.
    Você deverá completar as funcionalidades desta função
*/
ptrCliente criarNovoCliente(int idCliente, int numConta, float saldo) {
    ptrCliente novoCliente = (ptrCliente) malloc(sizeof(Cliente));
    if (novoCliente != NULL) {
        novoCliente->id = idCliente;
        novoCliente->conta.numero = numConta;
        novoCliente->conta.saldo = saldo;
        novoCliente->conta.saldoInicial = saldo; // se você tiver esse campo na struct Conta
        novoCliente->conta.movimentacoes = NULL; // Lista de extrato começa vazia
        novoCliente->prox = NULL;

        return novoCliente;
    }
    
    return NULL;
}

/**
    Função que tem por objetivo a criação de um novo banco.
    Lembre-se, o banco possui uma lista de clientes.
    Você deverá implementar as funcionalidades desta função
*/
Banco* criarBanco(char *nome) {
    // o retorno é apenas ilustrativo. Você deverá modificá-lo posteriormente para atender aos requisitos da aplicação.
    Banco *b = (Banco*) malloc(sizeof(Banco));
    if (b != NULL) {
        if (nome != NULL) {
            strncpy(b->nome, nome, MAX_NOME - 1); //eu fuxiquei alternativas a um laço de repetição e descobri esse comando. bem mais enxuto que um for/while
            b->nome[MAX_NOME - 1] = '\0'; 
        } else {
            b->nome[0] = '\0';
        }
        b->clientes = NULL;
    }
    return b;

}


/**
    Função que tem por objetivo liberar toda e qualquer memória alocada dinamicamente para o banco, na seguinte ordem:
    1 - Lista de movimentações de cada cliente
    2 - Lista de clientes
    3 - banco

    Você deverá implementar as funcionalidades desta função
*/
void liberarBanco(Banco *b) {
    if (b == NULL) return; //ve se tal banco existe
    
    ptrCliente cliAtual = b->clientes; //estrutura p percorrer lista
    while (cliAtual != NULL) { //expurga os clientes
        ptrCliente auxCliente = cliAtual;
        ptrMovimentacao movAtual = cliAtual->conta.movimentacoes;
        while (movAtual != NULL) { //expurga as movimentacoes
            ptrMovimentacao auxMov = movAtual; 
            movAtual = movAtual->prox; 
            free(auxMov); //tira o valor antecessor a movimentacao atual
}
cliAtual = cliAtual->prox;     
free(auxCliente);              
    }
    free(b); //expurga banco
}



// Função que realiza a abertura do arquivo. NÃO altere esta função
FILE* openFile(char *path, char *mode) {
    return fopen(path, mode);
}

/**
 * Função que realiza a leitura dos dados no arquivo.
 * Para uma melhor compreensão desta função, leia o arquivo LEIA-ME.txt

    -----------------------------------
    | ATENCAO: NÃO altere esta função |
    -----------------------------------

 * Colunas presentes no arquivo:
 * cn = Coluna n, onde n = (1..4)
 * c1 = mov
 * c2 = (idCliente ou TipoMov)
 * c3 = (numConta, idCliente ou idClienteOrig-idClienteDest)
 * c4 = valor
 * */
void readFile(FILE *ptr, Banco *banco)
{
    char c1[4], c3[8];
    int c2;
    int idCliente, idClienteOrig, idClienteDest, numConta;
    float c4, valor;
    // clientes usados para busca
    ptrCliente clienteOrig = NULL, clienteDest = NULL;

    // Iterando nas linhas do arquivo
    while (fscanf(ptr, "%s\t%d\t%s\t%f\n", c1, &c2, c3, &c4) != EOF) {
        valor = c4;

        if (strcmp("add", c1) == 0) { // adicionar cliente
            idCliente = c2;
            numConta = atoi(c3);


            ptrCliente novoCliente = criarNovoCliente(idCliente, numConta, valor);
            if (novoCliente) {
                 adicionarCliente(banco, novoCliente);
            } else {
                printf("ERRO - Ocorreu um erro ao tentar adicionar o cliente %d\n", idCliente);
                liberarBanco(banco);
                fclose(ptr);
                exit(1);
            }
        } else if (strcmp("mov", c1) == 0) { // realizar movimentacoes
            ptrMovimentacao novaMovimentacao = NULL;

            switch(c2) {
                case 0: // deposito
                    idCliente = atoi(c3);
                    clienteOrig = buscarCliente(banco, idCliente);

                    if (clienteOrig) {
                        novaMovimentacao = criarNovaMovimentacao(0, valor, clienteOrig->id, 0);

                        if (novaMovimentacao) {
                            realizarDeposito(clienteOrig, novaMovimentacao);
                            clienteOrig = NULL;
                        } else {
                            printf("ERRO - Ocorreu um erro ao tentar depositar %.2f para o cliente %d.\n", valor, idCliente);
                            liberarBanco(banco);
                            fclose(ptr);
                            exit(1);
                        }

                    } else {
                        printf("ERRO - Cliente %d nao encontrado para realizacao de deposito no valor de %.2f.\n", idCliente, valor);
                        liberarBanco(banco);
                        fclose(ptr);
                        exit(1);
                    }
                break;

                case 1: // saque
                    idCliente = atoi(c3);
                    clienteOrig = buscarCliente(banco, idCliente);

                    if (clienteOrig) {
                        novaMovimentacao = criarNovaMovimentacao(1, valor, clienteOrig->id, 0);

                        if (novaMovimentacao) {
                            realizarSaque(clienteOrig, novaMovimentacao);
                            clienteOrig = NULL;
                        } else {
                            printf("ERRO - Ocorreu um erro ao tentar realizar o saque de %.2f para o cliente %d.\n", valor, idCliente);
                            liberarBanco(banco);
                            fclose(ptr);
                            exit(1);
                        }

                    } else {
                        printf("ERRO - Ocorreu um erro ao tentar buscar o cliente %d para realizar o saque de %.2f.\n", idCliente, valor);
                        liberarBanco(banco);
                        fclose(ptr);
                        exit(1);
                    }
                break;

                case 2: // transferencia
                    idClienteOrig = atoi(strtok(c3, "-"));
                    idClienteDest = atoi(strtok(NULL, "-"));

                    clienteOrig = buscarCliente(banco, idClienteOrig);
                    clienteDest = buscarCliente(banco, idClienteDest);

                    if (clienteOrig && clienteDest) {
                        realizarTransferencia(clienteOrig, clienteDest, valor);
                        clienteOrig = NULL;
                        clienteDest = NULL;
                    } else {
                        printf("ERRO - Ocorreu um erro ao tentar buscar clientes orig %d e dest %d para transferencia do valor %.2f\n", idClienteOrig, idClienteDest, valor);
                        liberarBanco(banco);
                        fclose(ptr);
                        exit(1);
                    }
                break;

                default:
                    printf("ERRO - movimentacao desconhecida.\n");
                    liberarBanco(banco);
                    fclose(ptr);
                    exit(1);
                break;
            }
        }
    }
}

/**
    Função que tem por objetivo imprimir todos os dados do banco no padrão requerido da especificação
    Você deverá implementar as funcionalidades desta função
*/
void imprimirDados(Banco *b) {
    if (b == NULL) return;

    ptrCliente cli = b->clientes;
    while (cli != NULL) {
        printf("=====================================================\n");
        printf("Id. Cliente  : %d\n", cli->id);
        printf("Numero Conta : %d\n", cli->conta.numero);
        printf("Saldo inicial : %.2f\n", cli->conta.saldoInicial);
        printf("- - - - - - - - - - - - - - - - - - - Movimentacoes - - - - - - - - - - - - - - - - - - -\n");

        ptrMovimentacao mov = cli->conta.movimentacoes;
        while (mov != NULL) {
            if (mov->tipo == 0) { 
                printf("Tipo: Deposito | Valor: %.2f\n", mov->valor);
            } else if (mov->tipo == 1) {
                printf("Tipo: Saque    | Valor: -%.2f\n", mov->valor);
            } else if (mov->tipo == 2) {
                if (mov->idClienteOrig == cli->id) {
                    printf("Tipo: Transf. | Valor: -%.2f ===> Destinatario: %d\n", mov->valor, mov->idClienteDest);
                } else {
                    printf("Tipo: Transf. | Valor: %.2f ===> Origem: %d\n", mov->valor, mov->idClienteOrig);
                }
            }
            mov = mov->prox;
        }

        printf("Saldo Final: %.2f\n", cli->conta.saldo);
        cli = cli->prox;
    }
    printf("=====================================================\n");
}


int main(int argc, char *argv[])
{
    char path[MAX_PATH];
    FILE *filePtr = NULL;
    Banco *bomBanco = NULL;

    scanf("%s", path);
    filePtr = openFile(path, "r");

    if (filePtr) {
        bomBanco = criarBanco("BomBanco");
        if (bomBanco) {
            readFile(filePtr, bomBanco);
            imprimirDados(bomBanco);
            liberarBanco(bomBanco);
        }
        fclose(filePtr);
    } else {
        printf("Falha ao tentar abrir o arquivo\n");
        exit(1);
    }
    return 0;
}
