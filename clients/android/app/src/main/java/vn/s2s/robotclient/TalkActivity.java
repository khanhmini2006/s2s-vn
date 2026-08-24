package vn.s2s.robotclient;

import android.Manifest;
import android.app.Activity;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;

import vn.s2s.robotclient.audio.MicRecorder;
import vn.s2s.robotclient.audio.PlaybackBuffer;
import vn.s2s.robotclient.audio.SpeakerPlayer;
import vn.s2s.robotclient.net.RealtimeWsClient;

/**
 * Màn hình nói chuyện với server s2s-vn.
 *
 * <p>Robot chỉ đóng vai mic/loa từ xa: thu tiếng đẩy lên server, nhận audio trả về phát
 * ra loa. Toàn bộ VAD/STT/LLM/TTS chạy trên server.
 *
 * <p>Cách bấm: bấm một lần để mở mic, VAD của server tự nhận ra lúc người dùng nói xong
 * và client đóng mic khi nhận {@code input_audio_buffer.speech_stopped}. Mic không mở
 * thường trực — vừa tránh tranh mic với dịch vụ giọng nói của hãng, vừa tránh loa vọng
 * ngược vào mic làm VAD server tự kích.
 */
public class TalkActivity extends Activity implements RealtimeWsClient.Listener {

    private static final String PREFS = "s2s_robot_client";
    private static final String KHOA_URL = "url_server";
    /** IP LAN của máy dev chạy s2s-vn serve — sửa lại nếu đổi máy hoặc đổi mạng. */
    private static final String URL_MAC_DINH = "ws://10.1.50.41:8765/v1/realtime";
    private static final int MA_XIN_QUYEN_MIC = 1001;

    private EditText oNhapUrl;
    private Button nutNoi;
    private Button nutNoiChuyen;
    private TextView dongTrangThai;
    private TextView textHoiThoai;
    private TextView textLog;
    private ScrollView khungHoiThoai;
    private ScrollView khungLog;
    private vn.s2s.robotclient.ui.ThanhBienDo thanhBienDo;

    /** Màu trạng thái mang nghĩa: ngọc = ổn, hổ phách = đang nghe, đất nung = lỗi. */
    private static final int MAU_NGOC = 0xFF35C2B0;
    private static final int MAU_HO_PHACH = 0xFFFFB020;
    private static final int MAU_DAT_NUNG = 0xFFE05A47;
    private static final int MAU_KHOI = 0xFF8494A6;

    private final Handler uiHandler = new Handler(Looper.getMainLooper());
    private final PlaybackBuffer playbackBuffer = new PlaybackBuffer();
    private SpeakerPlayer speaker;
    private MicRecorder mic;
    private RealtimeWsClient ws;

    /** Câu trả lời đang được ghép dần từ các mẩu transcript. */
    private final StringBuilder cauDangGhep = new StringBuilder();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_talk);

        oNhapUrl = (EditText) findViewById(R.id.oNhapUrl);
        nutNoi = (Button) findViewById(R.id.nutNoi);
        nutNoiChuyen = (Button) findViewById(R.id.nutNoiChuyen);
        dongTrangThai = (TextView) findViewById(R.id.dongTrangThai);
        textHoiThoai = (TextView) findViewById(R.id.textHoiThoai);
        textLog = (TextView) findViewById(R.id.textLog);
        khungHoiThoai = (ScrollView) findViewById(R.id.khungHoiThoai);
        khungLog = (ScrollView) findViewById(R.id.khungLog);
        thanhBienDo = (vn.s2s.robotclient.ui.ThanhBienDo) findViewById(R.id.thanhBienDo);

        oNhapUrl.setText(getSharedPreferences(PREFS, MODE_PRIVATE)
                .getString(KHOA_URL, URL_MAC_DINH));

        nutNoi.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                batDauNoi();
            }
        });

        nutNoiChuyen.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                bamNutNoiChuyen();
            }
        });

        speaker = new SpeakerPlayer(playbackBuffer);
        mic = new MicRecorder(new MicRecorder.Listener() {
            @Override
            public void onChunk(byte[] pcm) {
                if (ws != null) {
                    ws.guiAudio(pcm);
                }
            }

            @Override
            public void onBienDo(int bienDoDinh) {
                thanhBienDo.themBienDo(bienDoDinh);
            }

            @Override
            public void onError(String thongDiep) {
                ghiLog("MIC LỖI: " + thongDiep);
                uiHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        datTrangThai("Lỗi mic", MAU_DAT_NUNG);
                        datNutNghe(false);
                    }
                });
            }
        });

        xinQuyenMicNeuCan();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mic != null) {
            mic.dungLai();
        }
        if (speaker != null) {
            speaker.dungLai();
        }
        if (ws != null) {
            ws.dong();
        }
    }

    /**
     * Xin quyền thu âm lúc chạy.
     *
     * <p>Bắt buộc từ API 23 — robot chạy Android 6.0.1. Khai trong manifest thôi thì
     * {@code AudioRecord} vẫn chạy nhưng trả toàn số 0 mà không báo lỗi gì.
     */
    private void xinQuyenMicNeuCan() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                        != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.RECORD_AUDIO}, MA_XIN_QUYEN_MIC);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,
            int[] grantResults) {
        if (requestCode == MA_XIN_QUYEN_MIC) {
            boolean duocPhep = grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED;
            ghiLog(duocPhep ? "đã được cấp quyền mic"
                    : "TỪ CHỐI quyền mic — sẽ không thu được tiếng");
        }
    }

    private void batDauNoi() {
        String url = oNhapUrl.getText().toString().trim();
        if (TextUtils.isEmpty(url)) {
            datTrangThai("Chưa nhập địa chỉ máy chủ", MAU_DAT_NUNG);
            return;
        }

        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KHOA_URL, url).apply();

        if (ws != null) {
            ws.dong();
        }
        ws = new RealtimeWsClient(url, this);

        datTrangThai("Đang nối", MAU_KHOI);
        ghiLog("nối tới " + url);
        speaker.batDau();
        ws.noi();
    }

    private void bamNutNoiChuyen() {
        if (ws == null || !ws.sanSangNoi()) {
            datTrangThai("Chưa sẵn sàng", MAU_KHOI);
            return;
        }

        if (mic.dangThu()) {
            // Bấm lần nữa để hủy giữa chừng (bình thường VAD server tự đóng mic).
            mic.dungLai();
            datNutNghe(false);
            datTrangThai("Sẵn sàng", MAU_NGOC);
            ghiLog("người dùng hủy lượt nói");
        } else {
            playbackBuffer.clear(); // im ngay câu cũ nếu robot đang đọc dở
            thanhBienDo.xoa();
            mic.batDau();
            datNutNghe(true);
            datTrangThai("Đang nghe", MAU_HO_PHACH);
            ghiLog("mở mic");
        }
    }

    // --- Sự kiện từ server (chạy ngoài luồng UI) ---

    @Override
    public void onConnected() {
        ghiLog("đã nối, chờ model nạp xong");
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                datTrangThai("Đang nạp mô hình", MAU_KHOI);
            }
        });
    }

    @Override
    public void onModelReady() {
        ghiLog("server.model_ready — sẵn sàng");
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                datTrangThai("Sẵn sàng", MAU_NGOC);
                nutNoiChuyen.setEnabled(true);
                datNutNghe(false);
            }
        });
    }

    @Override
    public void onSpeechStarted() {
        // VAD nghe thấy người nói: im ngay câu robot đang đọc (barge-in).
        playbackBuffer.clear();
        ghiLog("speech_started");
    }

    @Override
    public void onSpeechStopped() {
        // Người dùng nói xong — đóng mic, khỏi thu tiếng loa lúc robot trả lời.
        mic.dungLai();
        ghiLog("speech_stopped — đóng mic, chờ trả lời");
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                datTrangThai("Đang xử lý", MAU_KHOI);
                datNutNghe(false);
            }
        });
    }

    @Override
    public void onAudioChunk(byte[] pcm) {
        playbackBuffer.append(pcm);
    }

    @Override
    public void onTranscript(final String text) {
        cauDangGhep.append(text);
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                datTrangThai("Đang trả lời", MAU_NGOC);
            }
        });
    }

    @Override
    public void onResponseDone() {
        final String cau = cauDangGhep.toString().trim();
        cauDangGhep.setLength(0);
        ghiLog("response.done");
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                if (!cau.isEmpty()) {
                    themVaoHoiThoai("Robot: " + cau);
                }
                datTrangThai("Sẵn sàng", MAU_NGOC);
            }
        });
    }

    @Override
    public void onDisconnected(String lyDo, final int sePhutNoiLai) {
        ghiLog("mất kết nối (" + lyDo + "), nối lại sau " + sePhutNoiLai + "s");
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                nutNoiChuyen.setEnabled(false);
                nutNoiChuyen.setText("Đang chờ máy chủ");
                thanhBienDo.datDangThu(false);
                datTrangThai(sePhutNoiLai > 0
                        ? "Mất kết nối — nối lại sau " + sePhutNoiLai + "s"
                        : "Đã ngắt", MAU_DAT_NUNG);
            }
        });
    }

    @Override
    public void onError(String thongDiep) {
        ghiLog("LỖI: " + thongDiep);
    }

    // --- Tiện ích hiển thị ---

    private void datTrangThai(String s, int mau) {
        dongTrangThai.setText(s);
        dongTrangThai.setTextColor(mau);
    }

    /** Đổi nút và dải sóng giữa hai trạng thái: đang nghe / nghỉ. */
    private void datNutNghe(boolean dangNghe) {
        nutNoiChuyen.setSelected(dangNghe);
        nutNoiChuyen.setText(dangNghe ? "Đang nghe — chạm để dừng" : "Chạm để nói");
        thanhBienDo.datDangThu(dangNghe);
    }

    private void themVaoHoiThoai(final String dong) {
        textHoiThoai.append(dong + "\n\n");
        khungHoiThoai.post(new Runnable() {
            @Override
            public void run() {
                khungHoiThoai.fullScroll(View.FOCUS_DOWN);
            }
        });
    }

    /** Ghi log kỹ thuật; gọi được từ luồng bất kỳ. */
    private void ghiLog(final String dong) {
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                textLog.append(dong + "\n");
                khungLog.fullScroll(View.FOCUS_DOWN);
            }
        });
    }
}
