"use client";

import { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import { Upload, Play, Pause, ShieldCheck, ShieldAlert, Loader2, Activity } from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// --- CONFIGURATION ---
// ⚠️ REPLACE THIS WITH YOUR RENDER URL IF IT CHANGES
const API_URL = "https://deepvoice-api.onrender.com"; 

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurfer = useRef<WaveSurfer | null>(null);

  // --- AUDIO VISUALIZER LOGIC ---
  useEffect(() => {
    if (file && waveformRef.current) {
      // Initialize WaveSurfer
      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: "#4b5563",
        progressColor: "#3b82f6",
        cursorColor: "#60a5fa",
        barWidth: 2,
        barGap: 3,
        height: 100,
      });

      // Load audio
      const objectUrl = URL.createObjectURL(file);
      wavesurfer.current.load(objectUrl);

      // Cleanup
      return () => {
        wavesurfer.current?.destroy();
        URL.revokeObjectURL(objectUrl);
      };
    }
  }, [file]);

  const togglePlay = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      setIsPlaying(!isPlaying);
    }
  };

  // --- API CONNECTION LOGIC ---
  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("API Connection Failed");

      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Error analyzing file. Is the Backend running?");
      console.error(error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 p-8 font-mono">
      {/* HEADER */}
      <header className="max-w-4xl mx-auto mb-12 flex items-center justify-between border-b border-slate-800 pb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-600 rounded-lg">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">DeepVoice Guard</h1>
            <p className="text-slate-500 text-sm">M.Tech Grade Forensic Audio Analysis</p>
          </div>
        </div>
        <div className="text-xs text-slate-600 bg-slate-900 px-3 py-1 rounded-full border border-slate-800">
          v1.0.0 (TFLite Edge)
        </div>
      </header>

      <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* LEFT COLUMN: UPLOAD & VISUALIZER */}
        <div className="space-y-6">
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
            <h2 className="text-sm font-semibold text-slate-400 mb-4 uppercase tracking-wider">Input Signal</h2>
            
            {/* FILE UPLOADER */}
            <div className="relative">
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 flex flex-col items-center justify-center hover:border-blue-500 hover:bg-slate-800/50 transition-all">
                <Upload className="w-8 h-8 text-slate-500 mb-2" />
                <span className="text-sm text-slate-400">
                  {file ? file.name : "Drop audio file or click to upload"}
                </span>
              </div>
            </div>

            {/* WAVEFORM */}
            {file && (
              <div className="mt-6 space-y-4">
                <div ref={waveformRef} className="w-full" />
                <button
                  onClick={togglePlay}
                  className="flex items-center gap-2 text-xs font-bold text-blue-400 hover:text-blue-300"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isPlaying ? "PAUSE PREVIEW" : "PLAY PREVIEW"}
                </button>
              </div>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!file || isAnalyzing}
            className={twMerge(
              "w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all",
              !file 
                ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                : isAnalyzing
                ? "bg-blue-900/50 text-blue-200"
                : "bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20"
            )}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                PROCESSING SPECTROGRAM...
              </>
            ) : (
              "RUN DIAGNOSTICS"
            )}
          </button>
        </div>

        {/* RIGHT COLUMN: RESULTS */}
        <div className="space-y-6">
           <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 h-full flex flex-col">
            <h2 className="text-sm font-semibold text-slate-400 mb-4 uppercase tracking-wider">Analysis Report</h2>
            
            {!result ? (
              <div className="flex-1 flex flex-col items-center justify-center text-slate-600 space-y-4 opacity-50">
                <ShieldCheck className="w-16 h-16" />
                <p>System Ready. Awaiting Input.</p>
              </div>
            ) : (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                
                {/* VERDICT CARD */}
                <div className={clsx(
                  "p-6 rounded-xl border-2 flex items-center gap-4",
                  result.is_fake 
                    ? "bg-red-950/30 border-red-500/50 text-red-200" 
                    : "bg-emerald-950/30 border-emerald-500/50 text-emerald-200"
                )}>
                  {result.is_fake ? (
                    <ShieldAlert className="w-12 h-12 text-red-500" />
                  ) : (
                    <ShieldCheck className="w-12 h-12 text-emerald-500" />
                  )}
                  <div>
                    <h3 className="text-3xl font-bold">{result.label}</h3>
                    <p className="text-sm opacity-80">
                      Confidence: {(result.confidence * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>

                {/* DETAILS */}
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between p-3 bg-slate-950 rounded-lg">
                    <span className="text-slate-400">Engine</span>
                    <span className="font-mono text-blue-400">{result.engine}</span>
                  </div>
                  <div className="flex justify-between p-3 bg-slate-950 rounded-lg">
                    <span className="text-slate-400">Analysis Type</span>
                    <span className="font-mono text-purple-400">Mel-Spectrogram CNN</span>
                  </div>
                   <div className="flex justify-between p-3 bg-slate-950 rounded-lg">
                    <span className="text-slate-400">Filename</span>
                    <span className="font-mono text-slate-300 truncate max-w-[150px]">{result.filename}</span>
                  </div>
                </div>

              </div>
            )}
           </div>
        </div>

      </div>
    </main>
  );
}
