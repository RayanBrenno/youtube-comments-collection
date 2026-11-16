export type ClassificadorTipo = "encoder" | "claude";

export interface Classificador {
  id: string;
  nome: string;
  tipo: ClassificadorTipo;
  /** Acurácia e F1 macro no dataset 2 do TCC (500 comentários rotulados à mão). */
  acuracia_tcc: number;
  f1_macro_tcc: number;
  disponivel: boolean;
  /** Por que não está disponível (modelo ausente, chave não configurada...). */
  motivo: string | null;
}

export interface ClassificadoresResponse {
  padrao: string;
  classificadores: Classificador[];
}

/** Classificador usado em um vídeo, como o backend devolve junto com ele. */
export interface ClassificadorResumo {
  id: string;
  nome: string;
  tipo: ClassificadorTipo;
}
