import { type YoutubeComment } from "./comment";
import { type ClassificadorResumo } from "./classificador";

export interface YoutubeVideo {
  id?: string;
  code_url?: string;
  codeURL?: string;
  title: string;
  channel: string;
  publish_date?: string | null;
  views: number;
  likes: number;
  comments: number;
  thumbnail_url?: string | null;
  last_updated_at?: string | null;
  consulted_at?: string | null;
}

export interface ExistingVideo {
  id: string;
  code_url: string;
  title: string;
  channel: string;
  thumbnail_url?: string | null;
  last_updated_at?: string | null;
  consulted_at?: string | null;
  classificador?: ClassificadorResumo | null;
}

export interface HistoryVideo {
  idVideo: string;
  /** id do classificador usado nos comentários deste vídeo. */
  classificador?: string | null;
  code_url?: string;
  title: string;
  channel: string;
  publish_date?: string | null;
  views: number;
  likes: number;
  comments: number;
  thumbnail_url?: string | null;
  last_updated_at?: string | null;
  consulted_at?: string | null;
}

export interface PreviousVideoStats {
  views: number;
  likes: number;
  comments: number;
  last_updated_at?: string | null;
}

export interface VideoStatsDiff {
  views: number;
  likes: number;
  comments: number;
}

export interface CheckVideoResponse {
  exists: boolean;
  video: ExistingVideo | null;
}

export interface ConsultResponse {
  video: YoutubeVideo;
  classificador?: ClassificadorResumo | null;
  comments: YoutubeComment[];
  total_comments: number;
}

export interface UpdateVideoResponse extends ConsultResponse {
  previous?: PreviousVideoStats;
  diff?: VideoStatsDiff;
}

export interface UpdatedVideoViewState {
  before: PreviousVideoStats;
  after: UpdateVideoResponse;
}

export interface VideoResultCardProps {
  video: YoutubeVideo;
  totalComments?: number;
  previousStats?: PreviousVideoStats;
  classificador?: ClassificadorResumo | null;
}