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
 * <p>Cách bấm: bấm một lần để MỞ PHIÊN rồi hỏi liên tục nhiều lượt, không phải bấm lại
 * giữa các câu. Trong phiên, mic luân phiên: mở khi tới lượt người dùng, tạm nghỉ lúc
 * robot trả lời ({@code speech_stopped} → {@code mic.dungLai()}), tự mở lại sau
 * {@link #CHO_TAT_DUOI_TIENG_MS} kể từ {@code response.done}. Bấm lần nữa để đóng phiên.
 *
 * <p>Mic không mở xuyên suốt như {@code talk.py} vì loa robot vọng vào mic rất mạnh —
 * đo được biên độ chạm trần 32768 trong khi giọng người chỉ 1000–3400. Để mic mở lúc
 * loa phát thì VAD server nghe thấy tiếng của chính robot và chốt lượt rỗng liên tục.
 * Đổi lại, cắt lời bằng giọng không dùng được: muốn ngắt câu trả lời thì bấm nút
 * (nhánh mở phiên có {@code playbackBuffer.clear()} nên loa im ngay).
 */
public class TalkActivity extends Activity implements RealtimeWsClient.Listener {

    private static final String PREFS = "s2s_robot_client";
    private static final String KHOA_URL = "url_server";
    /** IP LAN của máy dev chạy s2s-vn serve — sửa lại nếu đổi máy hoặc đổi mạng. */
    private static final String URL_MAC_DINH = "ws://10.1.50.41:8765/v1/realtime";
    private static final int MA_XIN_QUYEN_MIC = 1001;

    /**
     * Chờ bấy nhiêu mili giây sau khi robot nói xong rồi mới mở lại mic.
     *
     * <p>Loa còn đuôi tiếng và tiếng vọng trong phòng; mở mic ngay thì thu lại chính
     * nó, VAD server tưởng có người nói và chốt một lượt rỗng.
     */
    private static final int CHO_TAT_DUOI_TIENG_MS = 600;

    /** Nhịp hỏi xem loa đọc xong chưa. */
    private static final int NHIP_KIEM_LOA_MS = 150;

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

    /**
     * Robot có đang phát tiếng không.
     *
     * <p>Bật khi nhận mẩu audio đầu tiên, tắt khi hết lượt. Không suy ra được từ
     * {@link SpeakerPlayer}: nó ghi im lặng liên tục để chống underrun nên lúc nào
     * cũng "đang phát".
     *
     * <p>Dùng để log phân biệt hai ca giống hệt nhau trên bề mặt: {@code speech_started}
     * do người nói chen vào (đúng ý), hay do mic nghe thấy tiếng loa của chính robot
     * (vọng âm, sẽ thành vòng lặp tự cắt lời chính mình).
     */
    private volatile boolean robotDangNoi;

    /**
     * Phiên hội thoại có đang mở không — do người dùng bật/tắt bằng nút.
     *
     * <p>Khác với {@code mic.dangThu()}: trong một phiên đang mở, mic vẫn tạm nghỉ mỗi
     * lúc robot nói rồi tự mở lại. Cờ này nhớ ý định của người dùng để biết có nên mở
     * lại hay không.
     */
    private volatile boolean phienDangMo;

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

        if (phienDangMo) {
            // Đóng phiên.
            phienDangMo = false;
            mic.dungLai();
            playbackBuffer.clear();
            datNutNghe(false);
            datTrangThai("Đã tắt mic", MAU_KHOI);
            ghiLog("đóng phiên");
        } else {
            // Mở phiên: hỏi liên tục nhiều lượt không cần bấm lại. Mic tạm nghỉ mỗi
            // lúc robot trả lời rồi tự mở lại — tránh thu tiếng loa của chính nó.
            phienDangMo = true;
            playbackBuffer.clear();
            thanhBienDo.xoa();
            mic.batDau();
            datNutNghe(true);
            datTrangThai("Đang nghe", MAU_HO_PHACH);
            ghiLog("mở phiên — hỏi liên tục, không cần bấm lại");
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
        // VAD nghe thấy tiếng: im ngay câu robot đang đọc (cắt lời).
        playbackBuffer.clear();
        ghiLog("speech_started");
    }

    @Override
    public void onSpeechStopped() {
        // Người dùng nói xong -> tạm đóng mic để khỏi thu tiếng loa lúc robot trả lời.
        // Phiên vẫn mở: mic tự bật lại ở onResponseDone, không cần bấm nút.
        mic.dungLai();
        ghiLog("speech_stopped — tạm nghỉ mic, chờ trả lời");
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                datTrangThai("Đang xử lý", MAU_KHOI);
                nutNoiChuyen.setText("Robot đang trả lời...");
                thanhBienDo.datDangThu(false);
            }
        });
    }

    @Override
    public void onAudioChunk(byte[] pcm) {
        robotDangNoi = true;
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
        robotDangNoi = false;
        final String cau = cauDangGhep.toString().trim();
        cauDangGhep.setLength(0);
        ghiLog("response.done");
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                if (!cau.isEmpty()) {
                    themVaoHoiThoai("Robot: " + cau);
                }

                if (phienDangMo) {
                    datTrangThai("Sắp nghe tiếp", MAU_KHOI);
                    choLoaDocXongRoiMoMic();
                } else {
                    datTrangThai("Sẵn sàng", MAU_NGOC);
                }
            }
        });
    }

    @Override
    public void onDisconnected(String lyDo, final int sePhutNoiLai) {
        ghiLog("mất kết nối (" + lyDo + "), nối lại sau " + sePhutNoiLai + "s");
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                phienDangMo = false; // mất kết nối thì đừng tự mở lại mic
                mic.dungLai();
                robotDangNoi = false;
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

    /**
     * Chờ loa đọc hết rồi mới mở lại mic.
     *
     * <p>Không mở mic theo hẹn giờ cố định kể từ {@code response.done}: sự kiện đó chỉ
     * báo server gửi xong dữ liệu, còn loa vẫn đang đọc phần tồn trong
     * {@link PlaybackBuffer}. Câu trả lời dài thì loa còn đọc thêm nhiều giây — mở mic
     * lúc ấy là thu ngay tiếng của chính robot (đo được biên độ chạm trần 32768, trong
     * khi giọng người chỉ 1000–3400).
     *
     * <p>Nên kiểm tra {@link PlaybackBuffer#conAudio()} theo nhịp; hết audio rồi mới
     * chờ thêm {@link #CHO_TAT_DUOI_TIENG_MS} cho đuôi tiếng và tiếng vọng trong phòng
     * lắng xuống.
     */
    private void choLoaDocXongRoiMoMic() {
        uiHandler.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (!phienDangMo || mic.dangThu()) {
                    return; // người dùng đã tắt phiên, hoặc mic mở rồi
                }
                if (playbackBuffer.conAudio()) {
                    uiHandler.postDelayed(this, NHIP_KIEM_LOA_MS); // loa còn đọc
                    return;
                }
                // Loa đã cạn — chờ nốt đuôi tiếng rồi mở mic.
                uiHandler.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        if (phienDangMo && !mic.dangThu() && !playbackBuffer.conAudio()) {
                            mic.batDau();
                            datNutNghe(true);
                            datTrangThai("Đang nghe", MAU_HO_PHACH);
                            ghiLog("loa đọc xong — mic mở lại, mời hỏi tiếp");
                        }
                    }
                }, CHO_TAT_DUOI_TIENG_MS);
            }
        }, NHIP_KIEM_LOA_MS);
    }

    /** Đổi nút và dải sóng giữa hai trạng thái: phiên đang mở / đã tắt. */
    private void datNutNghe(boolean dangNghe) {
        nutNoiChuyen.setSelected(dangNghe);
        nutNoiChuyen.setText(dangNghe ? "Đang nghe — chạm để dừng" : "Chạm để bắt đầu");
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
        // Ra logcat trước: khi app bị app khác cướp foreground hoặc màn hình không
        // xem được, đây là bản duy nhất còn lại để lấy qua adb.
        android.util.Log.i("TalkActivity", dong);
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                textLog.append(dong + "\n");
                khungLog.fullScroll(View.FOCUS_DOWN);
            }
        });
    }
}
